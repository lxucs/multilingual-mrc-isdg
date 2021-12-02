from os.path import join
from run_qa import QaRunner
import os
import sys
import json
from util import langs
import pickle


class Evaluator:
    def __init__(self, config_name, saved_suffix, gpu_id):
        self.saved_suffix = saved_suffix
        self.runner = QaRunner(config_name, gpu_id)
        self.model = self.runner.initialize_model(saved_suffix)

        self.output_dir = join(self.runner.config['log_dir'], 'results', saved_suffix)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_output_result_file(self, dataset_name, lang):
        return join(self.output_dir, f'results_{dataset_name}_{lang}.bin')

    def get_output_prediction_file(self, dataset_name, lang):
        return join(self.output_dir, f'prediction_{dataset_name}_{lang}.json')

    def get_output_nbest_file(self, dataset_name, lang):
        return join(self.output_dir, f'nbest_{dataset_name}_{lang}.json')

    def get_output_null_log_odds_file(self, dataset_name, lang):
        return join(self.output_dir, f'null_log_odds_{dataset_name}_{lang}.json')

    def get_output_metrics_file(self, dataset_name, lang):
        return join(self.output_dir, f'metrics_{dataset_name}_{lang}.json')

    def evaluate_task(self, dataset_name):
        assert dataset_name in langs

        all_f1, all_em = [], []
        for lang in langs[dataset_name]:
            examples, features, dataset = self.runner.data.get_target(dataset_name, 'test', lang)
            result_path = self.get_output_result_file(dataset_name, lang)

            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)
                metrics, _ = self.runner.evaluate_from_results(examples, features, results, verbose_logging=False,
                                                               dataset_name=dataset_name, lang=lang)
            else:
                metrics, _ = self.runner.evaluate(
                    self.model, examples, features, dataset, step=0, tb_writer=None,
                    output_results_file=self.get_output_result_file(dataset_name, lang),
                    output_prediction_file=self.get_output_prediction_file(dataset_name, lang),
                    output_nbest_file=self.get_output_nbest_file(dataset_name, lang),
                    output_null_log_odds_file=self.get_output_null_log_odds_file(dataset_name, lang),
                    dataset_name=dataset_name, lang=lang,
                    verbose_logging=False)

            print(f'Metrics for {dataset_name}-{lang}')
            for name, val in metrics.items():
                print(f'{name}: {val}')
            print('-' * 20)
            with open(self.get_output_metrics_file(dataset_name, lang), 'w') as f:
                json.dump(metrics, f)

            all_f1.append(metrics['f1'])
            all_em.append(metrics['exact_match'])

        print('-' * 20)
        print(f'Avg F1: %.4f' % (sum(all_f1) / len(all_f1)))
        print(f'Avg EM: %.4f' % (sum(all_em) / len(all_em)))


if __name__ == '__main__':
    config_name, saved_suffix, gpu_id, dataset_name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(dataset_name)
