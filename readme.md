# Raif

В рамках хакатона RaifHack мы делали оценку стоимости комерческой недвижиости. Сначала мы реализовали всё через baseline, то есть baseline идеей у нас было препроцессить данные, а потом использовать LGBM регрессию, с этим мы получили средненький скор. Второй идей мы написали небольшую нейронную сеть и обучили её на 100 эпохах, обучали батчами. Она работает неплохо и хорошо обучается, но к концу хакатона у нас возникла проблема с тестом на итоговом датасете, кстати данные мы в git не закидывали, потому что их объём слишком большой и не помещается в репозиторий. Нейронка была небольшая у неё всего лишь 3 слоя, но на данных для обучения мы получили очень хороший скор. Модель baseline, то есть через регрессию у нас находится в папке baseline/raif_hack и называется model_baseline.py, а модель с нейронкой в model.py. Инструкцию по запуску и тесту на данных вы можете посмотреть в файлах predict.py и train.py. Более подробно узнать про задачцу вы можете на сайте хакатона и посмотреть наш скор на регресси вы можете скорее всего там же, наша клманда killer coders.
Сайт хакатона: https://raifhack.ru/

Если вы хотите посмотреть нормальное решение через регрессию, которое дало хороший сокр, то посмотрите самый первый коммит комиит. 
