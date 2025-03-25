import hashlib

import numpy as np
import pandas as pd
import pytest

from generate_qwk_spiders import (
    determine_valid_n_reference_groups,
    determine_categories,
    calculate_kappas_and_intervals,
    calculate_delta_kappa,
    bootstrap_kappa,
    extract_ai_models,
    extract_plot_data,
    plot_spider_chart,
    bin_data,
    match_cases
)


# CDC age bin configuration
age_bins = {
    'age': {
        'bins': [0, 18, 30, 40, 50, 65, 75, 85, np.inf],
        'labels': ['<18', '18-29', '30-39', '40-49', '50-64', '65-74', '75-84', '85+']
    }
}

def dataframe_hash(df: pd.DataFrame) -> str:
    # Compute a hash for each row, including index in the hash.
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    # Convert the row hash Series into bytes.
    combined = row_hashes.values.tobytes()
    # Compute and return the MD5 hash of the combined bytes.
    return hashlib.md5(combined).hexdigest()


class TestGenerateQWKspiders:
    def setup_method(self):
        raw_df1 = pd.read_csv('../test_truthNdemographics.csv')
        df1 = bin_data(raw_df1, age_bins)
        df2 = pd.read_csv('../test_scores.csv')

        self.categories = determine_categories(df1)
        self.matched_df = match_cases(df1, df2)
        self.reference_groups, self.valid_groups, self.filtered_df = determine_valid_n_reference_groups(self.matched_df, self.categories)

    def test_determine_categories(self):
        expected = np.array(['age', 'ethnicity', 'race', 'sex', 'intersectional_race_ethnicity'])
        assert np.array_equal(self.categories, expected)

    def test_match_cases(self):
        assert isinstance(self.matched_df, pd.DataFrame)
        assert dataframe_hash(self.matched_df) == 'dfea59055cbfc865e1f7aed38e891e1e'

    def test_determine_validNreference_groups(self):
        reference_groups = self.reference_groups
        valid_groups = self.valid_groups
        filtered_df = self.filtered_df

        assert isinstance(reference_groups, dict)
        assert reference_groups == {
                                    'age': '50-64',
                                    'ethnicity': 'Not Hispanic or Latino',
                                    'intersectional_race_ethnicity': 'White and Not HispanicLatino',
                                    'race': 'White',
                                    'sex': 'Female',
                                    }
        assert isinstance(valid_groups, dict)
        assert valid_groups == {'age': {'18-29': np.int64(145),
                                        '30-39': np.int64(108),
                                        '40-49': np.int64(96),
                                        '50-64': np.int64(192),
                                        '65-74': np.int64(117),
                                        '75-84': np.int64(89),
                                        '85+': np.int64(33)},
                                'ethnicity': {'Hispanic or Latino': np.int64(157),
                                              'Not Hispanic or Latino': np.int64(637)},
                                'intersectional_race_ethnicity': {'Not White or Hispanic or Latino': np.int64(327),
                                                                  'White and Not HispanicLatino': np.int64(462)},
                                'race': {'Asian': np.int64(31),
                                         'Black or African American': np.int64(137),
                                         'Other': np.int64(77),
                                         'White': np.int64(541)},
                                'sex': {'Female': np.int64(436), 'Male': np.int64(378)}}
        assert isinstance(filtered_df, pd.DataFrame)
        assert dataframe_hash(filtered_df) == 'f25529198fad50242aa7733a6555f474'


    def test_calculate_kappas_and_intervals(self):
        np.random.seed(42)  # Set a fixed random seed for reproducibility
        ai_cols = ['ai_model_1', 'ai_model_2', 'ai_model_3']
        kappas, intervals = calculate_kappas_and_intervals(self.filtered_df, ai_cols)
        assert isinstance(kappas, dict)
        assert kappas == {'ai_model_1': np.float64(0.884169200407597),
                          'ai_model_2': np.float64(0.8763322198040202),
                          'ai_model_3': np.float64(0.8726076451226352)}
        assert isinstance(intervals, dict)
        assert intervals == {'ai_model_1': (np.float64(0.8600556307272443), np.float64(0.903832011444555)),
                             'ai_model_2': (np.float64(0.8531496446652914), np.float64(0.8949123673383993)),
                             'ai_model_3': (np.float64(0.8487998502873082), np.float64(0.89264456164595))}

    def test_bootstrap_kappa(self):
        ai_cols = ['ai_model_1', 'ai_model_2', 'ai_model_3']
        np.random.seed(42)  # Set a fixed random seed for reproducibility
        kappas1 = bootstrap_kappa(self.filtered_df, 'ai_model_1', n_iter=20)
        assert kappas1 == [np.float64(0.22905317769131006), np.float64(0.22514906878021268), np.float64(0.22882420983524565), np.float64(0.18861107389770515), np.float64(0.21474584801207852), np.float64(0.24085626176126795), np.float64(0.22623478875940617), np.float64(0.224892303538335), np.float64(0.22338042746773745), np.float64(0.23642662379101453), np.float64(0.23250395801031742), np.float64(0.22397443783464688), np.float64(0.2254233474328724), np.float64(0.21348187582861422), np.float64(0.22196094281042744), np.float64(0.1991649838324021), np.float64(0.1981408011969512), np.float64(0.21357311173485172), np.float64(0.22209395973154378), np.float64(0.2649499438981746)]
        np.random.seed(42)  # Set a fixed random seed for reproducibility
        kappas2 = bootstrap_kappa(self.filtered_df, 'ai_model_2', n_iter=20)
        assert kappas2 == [np.float64(0.2583893847852391), np.float64(0.2290516152959129), np.float64(0.22170349024690317), np.float64(0.2020987135424942), np.float64(0.22417039185894916), np.float64(0.22682961302970028), np.float64(0.2278065450899599), np.float64(0.24298770237088507), np.float64(0.23366065924944923), np.float64(0.23720234978635502), np.float64(0.23943933190092037), np.float64(0.23547919625965175), np.float64(0.24505066187617608), np.float64(0.22801820192917666), np.float64(0.25197971960109034), np.float64(0.21304075950243706), np.float64(0.21870292066303887), np.float64(0.25783399587481814), np.float64(0.23682068587994332), np.float64(0.26507044517393363)]
        np.random.seed(42)  # Set a fixed random seed for reproducibility
        kappas3 = bootstrap_kappa(self.filtered_df, 'ai_model_3', n_iter=20)
        assert kappas3 == [np.float64(0.1101532187821681), np.float64(0.07629487919583422), np.float64(0.07764277541242703), np.float64(0.060717659648696354), np.float64(0.09797380121638632), np.float64(0.08860286928232386), np.float64(0.0671393996343953), np.float64(0.0830619953755718), np.float64(0.09895600577218744), np.float64(0.06657368113877593), np.float64(0.0899968691604427), np.float64(0.09340608270725748), np.float64(0.11067620009149559), np.float64(0.07466336657039208), np.float64(0.09768156847925158), np.float64(0.09049434620450159), np.float64(0.07304935324991169), np.float64(0.10084297390137009), np.float64(0.09444831987063129), np.float64(0.10184078744245695)]

    delta_kappas_expected = {'ethnicity': {'ai_model_1': {'Hispanic or Latino': (np.float64(0.001278662238348549),
                                                                                 (np.float64(-0.06888012867844723),
                                                                                  np.float64(0.07628497812662427)))},
                                           'ai_model_2': {'Hispanic or Latino': (np.float64(-0.013730042671233567),
                                                                                 (np.float64(-0.08285091561497487),
                                                                                  np.float64(0.06444885857020462)))},
                                           'ai_model_3': {'Hispanic or Latino': (np.float64(-0.003157245818484977),
                                                                                 (np.float64(-0.06508193505521057),
                                                                                  np.float64(0.06026942619803746)))}},
                             'race': {'ai_model_1': {'Asian': (np.float64(0.09763877928598996),
                                                               (np.float64(-0.06285315596621108),
                                                                np.float64(0.2513377531908344))),
                                                     'Black or African American': (np.float64(0.07376660095900744),
                                                                                   (np.float64(-0.012405334180939925),
                                                                                    np.float64(0.15907592516420985))),
                                                     'Other': (np.float64(0.09845333153185265),
                                                               (np.float64(-0.004293903755987372),
                                                                np.float64(0.20901139643242286)))},
                                      'ai_model_2': {'Asian': (np.float64(0.04479319487605071),
                                                               (np.float64(-0.10592624507267234),
                                                                np.float64(0.18710734742724144))),
                                                     'Black or African American': (np.float64(0.04174201451931148),
                                                                                   (np.float64(-0.04346980199753037),
                                                                                    np.float64(0.12328220687270819))),
                                                     'Other': (np.float64(0.027912839622554975),
                                                               (np.float64(-0.07016876828086482),
                                                                np.float64(0.12976900751021073)))},
                                      'ai_model_3': {'Asian': (np.float64(0.1450184080257732),
                                                               (np.float64(0.011697736617909679),
                                                                np.float64(0.3019585745923135))),
                                                     'Black or African American': (np.float64(0.004653947642248857),
                                                                                   (np.float64(-0.05626765254885206),
                                                                                    np.float64(0.0791364427079219))),
                                                     'Other': (np.float64(0.05011601775038271),
                                                               (np.float64(-0.04383938125140975),
                                                                np.float64(0.14768272493996135)))}}}

    def test_calculate_delta_kappa(self):
        skip_test = False
        if skip_test:  # Skip this test to save time
            return
        categories = ['ethnicity', 'race']
        reference_groups = {
            'ethnicity': 'Not Hispanic or Latino',
            'race': 'White',
        }
        ai_cols = ['ai_model_1', 'ai_model_2', 'ai_model_3']
        np.random.seed(42) # Set a fixed random seed for reproducibility
        delta_kappas = calculate_delta_kappa(self.filtered_df, categories, reference_groups, ai_cols)
        assert isinstance(delta_kappas, dict)
        assert delta_kappas == TestGenerateQWKspiders.delta_kappas_expected

    def test_extract_plot_data(self):
        model = 'ai_model_1'
        groups, values, lower_bounds, upper_bounds = extract_plot_data(TestGenerateQWKspiders.delta_kappas_expected, model)
        assert groups == ['ethnicity: Hispanic or Latino', 'race: Asian', 'race: Black or African American', 'race: Other']
        assert values == [np.float64(0.001278662238348549), np.float64(0.09763877928598996), np.float64(0.07376660095900744), np.float64(0.09845333153185265)]
        assert lower_bounds == [np.float64(-0.06888012867844723), np.float64(-0.06285315596621108), np.float64(-0.012405334180939925), np.float64(-0.004293903755987372)]
        assert upper_bounds == [np.float64(0.07628497812662427), np.float64(0.2513377531908344), np.float64(0.15907592516420985), np.float64(0.20901139643242286)]

