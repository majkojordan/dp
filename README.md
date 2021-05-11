# Inštalačná príručka

## Požiadavky

Je potrebné mať Python 3 prostredie. Je možné využiť Conda prostredie, nasledovné príkazy je v tom prípade nutné prispôsobiť.

## Použitie

1. (VOLITEĽNÉ) vytvorte a aktivujte si virtuálne prostredie:
   ```
   python3 -m venv env && source env/bin/activate
   ```
1. nainštalujte potrebné knižnice:
   ```
   pip3 install -r requirements.txt
   ```
1. nastavte [konfiguráciu](#konfigurácia)
1. spustite `main.py`:
   ```
   python3 main.py
   ```

## Konfigurácia

Konfigurácia sa nastavuje pomocou premenných prostredia, odporúčané je využitie `.env` súboru umiestneného v základnom priečinku. V súbore `.env.example` sa nachádza vzorová konfigurácia.

- `DB_CONNECTION_STRING`: (string) connection string na pripojenie k databáze s originálnymi dátami
- `BASE_PATH`: (string) cesta k základnému priečinku
- `DATA_DIR`: (string) cesta k priečinku s dátami
- `DATASET`: (string) názov dátovej sady
- `DEBUG`: (boolean) debug mód
- `DEBUG_FOLDER`: (string) cesta k priečinku na ukladanie debug výstupov
- `BATCH_SIZE`: (int) veľkosť vstupnej vzorky neurónovej siete
- `EPOCHS`: (int) počet trénovacích epôch
- `HIDDEN_SIZE`: (int) veľkosť skrytej vrstvy neurónovej siete
- `EMBEDDING_SIZE`: (int) veľkosť Word2vec vektorov
- `WINDOW_SIZE`: (int) veľkosť trénovacieho okna Word2vec vektorov
- `MAX_VALIDATION_SIZE`: (int) maximálny počet sedení vo validačnej vzorke
- `MAX_TEST_SIZE`: (int) maximálny počet sedení v testovacej vzorke
- `NUM_LAYERS`: (int) počet skrytých vrstiev neurónovej siete
- `MANUAL_SEED`: (int) manuálna seed hodnota pre náhodnú funkciu
- `DETECT_PREFERENCE_CHANGE`: (int) metóda detekcie zmeny preferencie
  - 0 - žiadna
  - 1 - oddeľovanie
  - 2 - filtrovanie
- `LEARNING_RATE`: (float <0, 1>) rýchlosť učenia
- `SIMILARITY_THRESHOLD`: (float <0, 1>) minimálna hodnota podobnosti na detegovanie zmeny preferencie
- `INPUT_DROPOUT`: (float <0, 1>) dropout vstupnej vrstvy
- `HIDDEN_DROPOUT`: (float <0, 1>) dropout skrytých vrstiev (okrem poslednej)
- `USE_CATEGORY_SIMILARITY`: (boolean) využitie kategorickej podobnosti namiesto podobnosti samotných položiek
- `EVALUATE_MODEL`: (boolean) evaluačný mód
- `EVAL_FOLDER`: (string) cesta k priečinku na uloženie výsledkov evaluácie
- `SAVE_MODEL`: (boolean) uloženie natrénovaných modelov
- `SAVE_FOLDER`: (string) cesta k priečinku na uloženie natrénovaných modelov
- `HYBRID_ORIGINAL_MODEL_PATH`: (string) cesta k modelu, ktorý sa využije na vytvorenie odporúčaní pre neupravené sedenia
- `MODIFY_TRAIN`: (boolean) prispôsobenie preferencie aj v trénovacej množine
