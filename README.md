# preprocessing-ivy-data-ozone

## without normalization
```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2
```

## with normalization
```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --normalization min-max
```

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --normalization adjusted-min-max
```

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --normalization min-max-chunk
```

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --normalization z-score-chunk
```

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --normalization z-score
```

# data-augmention
```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut
```

```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization min-max
```