import torch
from tqdm import tqdm

from lib.utils.model import save_model
from lib.logger.debug import DebugLogger
from lib.logger.eval import EvalLogger
from lib.utils.common import print_line_separator


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        dataloaders,
        dataset,
        original_model,
        device=torch.device("cpu"),
        debug=False,
        evaluate=False,
        save=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.dataset = dataset
        self.original_model = original_model
        self.device = device
        self.debug = debug
        self.evaluate = evaluate
        self.save = save

    def train_phase(self, name, dataloader, is_train, debugLogger, evalLogger):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(is_train):
            hits = 0
            hits_10 = 0
            long_hits_10 = 0
            long_session_count = 0
            running_loss = 0

            for inputs, labels, metadata in tqdm(dataloader):
                inputs = inputs.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)
                session_ids, lengths = zip(*metadata)

                self.optimizer.zero_grad()
                self.model.reset_hidden()

                curr_batch_size = inputs.shape[0]
                if is_train:
                    y_pred = self.model(inputs, lengths)

                    loss = self.loss_fn(y_pred, labels)
                    loss.backward()
                    self.optimizer.step()
                else:
                    if self.original_model:
                        # hybrid mode - predict modified sessions with modified model and original sessions with original model
                        modified_indexes = [
                            i in self.dataset.modified_session_ids for i in session_ids
                        ]
                        original_indexes = [not i for i in modified_indexes]

                        self.original_model.reset_hidden()
                        y_pred_original = self.original_model(
                            inputs[original_indexes],
                            torch.Tensor(lengths)[original_indexes].tolist(),
                        )

                        y_pred_modified = self.model(
                            inputs[modified_indexes],
                            torch.Tensor(lengths)[modified_indexes].tolist(),
                        )

                        # merge the results
                        y_pred = []
                        original_i = 0
                        modified_i = 0
                        for is_modified in modified_indexes:
                            if is_modified:
                                y_pred.append(y_pred_modified[modified_i])
                                modified_i += 1
                            else:
                                y_pred.append(y_pred_original[original_i])
                                original_i += 1

                        y_pred = torch.stack(y_pred)

                    else:
                        y_pred = self.model(inputs, lengths)

                    loss = self.loss_fn(y_pred, labels)
                    # calculate hits@10 - only for test as it would slow down training
                    predicted_indexes_10 = torch.topk(y_pred, 10, axis=1).indices
                    hits_10 += sum(
                        [
                            l in predicted_indexes_10[i]
                            for i, l in enumerate(labels.tolist())
                        ]
                    )

                    # show the predictions
                    if debugLogger:
                        debugLogger.log(
                            dataset=self.dataset,
                            session_ids=session_ids,
                            predicted_indexes=predicted_indexes_10,
                        )

                    # log prediction result
                    if evalLogger:
                        evalLogger.log(
                            session_ids=session_ids,
                            predicted_indexes=predicted_indexes_10,
                            labels=labels.tolist(),
                        )

                    # calculate hits@10 for long sessions only
                    long_indexes = [
                        i in self.dataset.long_session_ids for i in session_ids
                    ]
                    long_inputs = inputs[long_indexes]
                    long_labels = labels[long_indexes]
                    long_preds = y_pred[long_indexes]

                    predicted_indexes_10 = torch.topk(long_preds, 10, axis=1).indices
                    long_hits_10 += sum(
                        [
                            l in predicted_indexes_10[i]
                            for i, l in enumerate(long_labels.tolist())
                        ]
                    )
                    long_session_count += long_inputs.shape[0]

                predicted_indexes = torch.argmax(y_pred, 1)
                hits += torch.sum(predicted_indexes == labels).item()
                running_loss += loss.item() * curr_batch_size

        phase_size = len(dataloader.dataset)
        if phase_size == 0:
            print("No sessions")
            return
        avg_loss = running_loss / phase_size
        acc = hits / phase_size * 100
        if is_train:
            print(f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}\n")
        else:
            acc_10 = hits_10 / phase_size * 100
            long_acc_10 = (
                long_hits_10 / long_session_count * 100
                if long_session_count > 0
                else 999
            )
            print(
                f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}, acc@10: {acc_10:.4f}, long acc@10: {long_acc_10:.4f}\n"
            )

            if evalLogger:
                evalLogger.write(name)
                evalLogger.reset()

        print_line_separator()

    def train_epoch(self, name):
        debugLogger = DebugLogger(name) if self.debug else None
        evalLogger = EvalLogger(name) if self.evaluate else None

        for phase, dataloader in self.dataloaders.items():
            print(f"Phase: {phase}")

            is_train = phase == "train"

            self.train_phase(
                name=phase,
                dataloader=dataloader,
                is_train=is_train,
                debugLogger=debugLogger,
                evalLogger=evalLogger,
            )

    def train(self, epochs=10):
        print(f"Training")

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch} / {epochs}\n")

            self.train_epoch(epoch)

            if self.save:
                save_model(self.model, epoch)
