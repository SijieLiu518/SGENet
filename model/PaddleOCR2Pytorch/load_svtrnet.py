from model.PaddleOCR2Pytorch.tools.infer.predict_rec import TextRecognizer
import model.PaddleOCR2Pytorch.tools.infer.pytorchocr_utility as utility


def load_svtrnet(parser):
    args = utility.parse_args(parser)
    args.rec_model_path = './workdir/rec_svtr_tiny_none_ctc_en_infer.pth'
    args.rec_algorithm = 'SVTR'
    args.rec_yaml_path = './model/PaddleOCR2Pytorch/configs/rec/rec_svtr/rec_svtr_tiny_6local_6global_stn_en.yml'
    text_recognizer = TextRecognizer(args)
    print(text_recognizer.net)
    return text_recognizer.net