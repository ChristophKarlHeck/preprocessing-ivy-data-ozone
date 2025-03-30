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

## without normalization
```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut
```

## with normalization
```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization min-max
```

```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization adjusted-min-max
```

```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization min-max-chunk
```

```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization z-score-chunk
```

```bash
python3 data_augmention.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut --normalization z-score
```

# Make simulation data

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2 --create-simulation-files 1
```

```bash
python3 preprocess.py --data-dir /home/chris/experiment_data/ozone_cut/ozone_cut/Exp45_Ivy4 --create-simulation-files 1
```