import cst_airfoil_codec
import util
from data import cst_airfoil_code_cache


def test_validate_cache_requires_finite_fit_metrics():
    identity = {
        'encoding_version': 'test',
        'source_sha256': 'test',
        'fit_config': {},
    }
    key = cst_airfoil_code_cache.cache_key(identity)
    cache = {
        'schema': cst_airfoil_code_cache.CACHE_SCHEMA,
        'entries': {
            key: {
                'identity': identity,
                'source_airfoil': 'test.dat',
                'code': [0.0] * util.CST_AIRFOIL_CODE_SIZE,
                'metrics': {
                    metric_name: 0.0
                    for metric_name in cst_airfoil_codec.CST_FIT_METRIC_KEYS
                },
            },
        },
    }

    cst_airfoil_code_cache.validate_cache(cache)

    del cache['entries'][key]['metrics']['mae']
    try:
        cst_airfoil_code_cache.validate_cache(cache)
    except KeyError as error:
        assert "missing metric 'mae'" in str(error)
    else:
        raise AssertionError('Expected missing fit metrics to invalidate the cache')


def test_validate_cache_supports_non_default_shape_coefficient_count():
    coefficient_count = 8
    identity = {
        'encoding_version': 'test',
        'source_sha256': 'test',
        'fit_config': {},
        'shape_coefficient_count': coefficient_count,
    }
    key = cst_airfoil_code_cache.cache_key(identity)
    cache = {
        'schema': cst_airfoil_code_cache.CACHE_SCHEMA,
        'entries': {
            key: {
                'identity': identity,
                'source_airfoil': 'test.dat',
                'code': [0.0] * cst_airfoil_codec.cst_airfoil_code_size(coefficient_count),
                'metrics': {
                    metric_name: 0.0
                    for metric_name in cst_airfoil_codec.CST_FIT_METRIC_KEYS
                },
            },
        },
    }

    cst_airfoil_code_cache.validate_cache(cache)
