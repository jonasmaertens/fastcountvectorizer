This package contains a CountVectorizer that is based on the scikit-learn CountVectorizer but is reduced to the most important functionality and optimized for performance. Vectorization speed is about double the speed of the scikit-learn CountVectorizer. 

### Build
Build the package:
```bash
pip install build
python -m build
```
Install the wheel from the dist directory:
```bash
pip install dist/fastcountvectorizer-{something}.whl
```
Or install the package directly from the .tar.gz in the release:
```bash
pip install fastcountvectorizer-{something}.tar.gz
```

### Usage
Import FastCountVectorizer
```python
from fastcountvectorizer import FastCountVectorizer
```
Usage of the FastCountVectorizer:
```python
vectorizer = FastCountVectorizer(binary=True, min_df=2, dtype=np.float32)
x_train = vectorizer.fit_transform(train_df["tokens"])
x_test = vectorizer.transform(test_df["tokens"])
```

