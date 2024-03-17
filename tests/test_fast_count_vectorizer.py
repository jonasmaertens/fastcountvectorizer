import unittest

from fastcountvectorizer import FastCountVectorizer


class TestFastCountVectorizer(unittest.TestCase):
    def setUp(self):
        self.train_dac = [
            "a ab abc abdfef <ab=dCf> f",
            "f f f aa ab abc abdfef <ab=dCf> f",
            "a",
            "a a",
            "<ab=dCf>",
        ]
        self.test_dac = [
            "a ab abc abdfef <ab=dCf> f",
            "a outOfVoc",
            "a a",
            "<ab=dCf> alsoOutOfVoc alsoOutOfVoc a",
        ]

    def test_fit(self):
        cv = FastCountVectorizer()
        cv.fit(self.train_dac)
        self.assertEqual(cv.vocabulary_, {'a': 0, 'ab': 1, 'abc': 2, 'abdfef': 3, '<ab=dCf>': 4, 'f': 5, 'aa': 6})

    def test_transform_non_binary(self):
        cv = FastCountVectorizer(binary=False)
        cv.fit(self.train_dac)
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0]
        ])

    def test_transform_binary(self):
        cv = FastCountVectorizer(binary=True)
        cv.fit(self.train_dac)
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0]
        ])

    def test_transform_binary_with_vocabulary(self):
        cv = FastCountVectorizer(binary=True, vocabulary={'a': 0, '<ab=dCf>': 1, 'f': 2})
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])

    def test_transform_non_binary_with_vocabulary(self):
        cv = FastCountVectorizer(binary=False, vocabulary={'a': 0, '<ab=dCf>': 1, 'f': 2})
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1, 1],
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0]
        ])

    def test_fit_transform_binary(self):
        cv = FastCountVectorizer(binary=True)
        transformed = cv.fit_transform(self.train_dac)
        self.assertEqual(transformed.toarray().tolist(), [
            [1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]
        ])
        self.assertEqual(cv.vocabulary_, {'a': 0, 'ab': 1, 'abc': 2, 'abdfef': 3, '<ab=dCf>': 4, 'f': 5, 'aa': 6})

    def test_fit_transform_non_binary(self):
        cv = FastCountVectorizer(binary=False)
        transformed = cv.fit_transform(self.train_dac)
        self.assertEqual(transformed.toarray().tolist(), [
            [1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 4, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]
        ])
        self.assertEqual(cv.vocabulary_, {'a': 0, 'ab': 1, 'abc': 2, 'abdfef': 3, '<ab=dCf>': 4, 'f': 5, 'aa': 6})

    def test_fit_min_df(self):
        cv = FastCountVectorizer(min_df=3)
        cv.fit(self.train_dac)
        self.assertEqual(cv.vocabulary_, {'a': 0, '<ab=dCf>': 1})

    def test_transform_non_binary_min_df(self):
        cv = FastCountVectorizer(binary=False, min_df=3)
        cv.fit(self.train_dac)
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1],
            [1, 0],
            [2, 0],
            [1, 1]
        ])

    def test_transform_binary_min_df(self):
        cv = FastCountVectorizer(binary=True, min_df=3)
        cv.fit(self.train_dac)
        self.assertEqual(cv.transform(self.test_dac).toarray().tolist(), [
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 1]
        ])

    def test_fit_transform_binary_min_df(self):
        cv = FastCountVectorizer(binary=True, min_df=3)
        transformed = cv.fit_transform(self.train_dac)
        self.assertEqual(transformed.toarray().tolist(), [
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1]
        ])
        self.assertEqual(cv.vocabulary_, {'a': 0, '<ab=dCf>': 1})

    def test_fit_transform_non_binary_min_df(self):
        cv = FastCountVectorizer(binary=False, min_df=3)
        transformed = cv.fit_transform(self.train_dac)
        self.assertEqual(transformed.toarray().tolist(), [
            [1, 1],
            [0, 1],
            [1, 0],
            [2, 0],
            [0, 1]
        ])
        self.assertEqual(cv.vocabulary_, {'a': 0, '<ab=dCf>': 1})

    def test_partial_fit_and_limit_features(self):
        cv = FastCountVectorizer(binary=True, min_df=3)
        cv.partial_fit(["a b c d e"])
        self.assertEqual(cv.vocabulary_, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})
        cv.partial_fit(["a b d e f"])
        self.assertEqual(cv.vocabulary_, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5})
        cv.partial_fit(["a e m b p"])
        self.assertEqual(cv.vocabulary_, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'm': 6, 'p': 7})
        cv.limit_features()
        self.assertEqual(cv.vocabulary_, {'a': 0, 'b': 1, 'e': 2})
