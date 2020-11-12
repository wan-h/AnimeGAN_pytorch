# coding: utf-8
# Author: wanhui0729@gmail.com


import torch
import logging
from animeganv2.utils.comm import is_main_process, get_world_size, synchronize



def inference(
        model,
        data_loader,
        device="cuda",
        output_folder=None,
        logger_name=None,
        evaluate_type='voc',
        evaluate_fn=None,
        **kwargs
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger(logger_name + ".inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(data_loader.dataset.__class__.__name__, len(dataset)))
    predictions, total_time = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions, logger_name)
    if not is_main_process():
        return None

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    evaluate_fn = evaluate_fn or evaluate

    return evaluate_fn(dataset=dataset,
                       predictions=predictions,
                       output_folder=output_folder,
                       logger_name=logger_name,
                       evaluate_type=evaluate_type,
                       **kwargs)

class Evaluator(object):
    def __init__(self, dataset, device="cuda", output_folder=None, logger_name=None):
        self.dataset = dataset
        self.device = device
        self.logger_name = logger_name
        self.output_folder = output_folder

    def do_inference(self, model):
        result = inference(
                    model=model,
                    data_loader=self.data_loader,
                    device=self.device,
                    output_folder=self.output_folder,
                    logger_name=self.logger_name,
                    evaluate_type=self.evaluate_type,
                    evaluate_fn=self.evaluate_fn,
                    **self.kwargs)
        return result