from overrides import overrides
from allennlp.training.metrics.metric import Metric
import os
from benchmark import process_outputs, carb
import json

@Metric.register("carb_doc")
class Carb(Metric):
    """
    Computes scores according to carb framework
    """
    def __init__(self, dev_set: str = None):
        super(Carb, self).__init__()
        self._all_predictions = []
        self._all_confidences = []
        self._all_example_doc_ids = []
        self._all_example_ids = []
        self._dev_set = dev_set
        self._epoch_num = 0


    def __call__(self, predictions: list, confidences: list, example_doc_ids: list, example_ids: list):
        self._all_predictions.extend(predictions)
        self._all_confidences.extend(confidences)
        self._all_example_doc_ids.extend(example_doc_ids)
        self._all_example_ids.extend(example_ids)


    def get_metric(self, reset: bool = False):
        if reset:
            self._epoch_num += 1
            dev_sents_file = ''
            if self._dev_set == 'dev_health':
                dev_sents_file = os.path.abspath('data/dev/healthcare/annotation_healthcare_400_doc.json')
            elif self._dev_set == 'dev_transport':
                dev_sents_file = os.path.abspath('data/dev/transport/annotation_traffic_400_doc.json')


            file = open(dev_sents_file, 'r', encoding='utf-8')
            data = json.load(file)
            doc_list = []
            for doc_num, doc in enumerate(data):
                sent_list = []
                for sent_num, sent in enumerate(data[doc]):
                    sent_list.extend([sent])
                doc_list.append(sent_list)

            input_lines = []
            for pred, conf in zip(self._all_predictions, self._all_confidences):
                json_acceptable_str = [[token.replace("'", "\'").replace('"', '\\"') for token in p] for p in pred]
                d = {}
                d["predicted_tokens"] = json_acceptable_str
                d["predicted_log_probs"] = conf.tolist()
                input_lines.append(json.dumps(d))

            # reorder dev sents according to the order of prediction
            dev_sents = [doc_list[example_doc_id][all_example_id]
                         for example_doc_id, all_example_id in zip(self._all_example_doc_ids, self._all_example_ids)]

            output_lines = process_outputs.process_single(input_lines, dev_sents, threshold=None)

            matchingFunc = carb.Matcher.binary_linient_tuple_match
            if self._dev_set == 'dev_health':
                dev_pred_file = os.path.abspath('data/dev/healthcare/extractions.tsv')
            elif self._dev_set == 'dev_transport':
                dev_pred_file = os.path.abspath('data/test/transport/extractions.tsv')

            if output_lines.strip() == '':
                return {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

            # evaluate outputs using carb
            predicted = carb.AllennlpReader(threshold = None)
            predicted.read(output_lines)

            b = carb.Benchmark(dev_pred_file)
            out_filename = "/dev/null"

            auc, optimal_f1_point, _ = b.compare(predicted=predicted.oie, matchingFunc=matchingFunc,
                                    output_fn=out_filename, error_file=None, binary=False)
            print("AUC: {}\t Optimal (precision, recall, F1): {}".format( auc, optimal_f1_point ))
            self.reset()
            return {'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_sum': (auc+optimal_f1_point[2])}

        else:
            return {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    @overrides
    def reset(self):
        self._all_predictions = []
        self._all_confidences = []
        self._all_example_doc_ids = []
        self._all_example_ids = []
