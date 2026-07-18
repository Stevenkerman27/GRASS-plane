from pathlib import Path

import cst_airfoil_codec
from data import precompute_airfoil_codes


def test_build_fit_report_records_all_samples_and_top_errors():
    first_metrics = {
        metric_name: float(index + 1)
        for index, metric_name in enumerate(cst_airfoil_codec.CST_FIT_METRIC_KEYS)
    }
    second_metrics = {
        metric_name: float(index + 2)
        for index, metric_name in enumerate(cst_airfoil_codec.CST_FIT_METRIC_KEYS)
    }
    first_entry = {'metrics': first_metrics}
    second_entry = {'metrics': second_metrics}
    entries = [(Path('first.dat'), first_entry), (Path('second.dat'), second_entry)]
    errors = precompute_airfoil_codes.top_error_entries(entries, count=1)

    report = precompute_airfoil_codes.build_fit_report(entries, errors)

    assert report['schema'] == precompute_airfoil_codes.FIT_REPORT_SCHEMA
    assert report['sample_count'] == 2
    assert report['metric_summary']['mae'] == {'min': 1.0, 'mean': 1.5, 'max': 2.0}
    assert report['largest_max_point_errors'][0]['source_airfoil'].endswith('second.dat')
    assert len(report['samples']) == 2
