import clip
import numpy as np
import torch

VID_classes = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']

VIS_classes = [
        "__background__",  # From VID. TODO: is this required?
        "person", "giant panda", "lizard", "parrot", "skateboard", "sedan", "ape", "dog", "snake",
        "monkey", "hand", "rabbit", "duck", "cat", "cow", "fish", "train", "horse", "turtle",
        "bear", "motorbike", "giraffe", "leopard", "fox", "deer", "owl", "surfboard", "airplane",
        "truck", "zebra", "tiger", "elephant", "snowboard", "boat", "shark", "mouse", "frog",
        "eagle", "earless seal", "tennis racket"
    ]

def get_text_prompts(clip_model, prompt, device):
    with torch.no_grad():
        text = clip.tokenize(prompt).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


template_sentence_list = ["a photo of %s"]

VID_classes_dict = {'__background__': [],  # always index 0
                    'airplane':["plane", "aeroplane", "airliner", "aircraft", "jetliner"],
                    'antelope':["gazelle", "impala", "Aepyceros melampus"],
                    'bear':["Ursidae", "black bear", "brown bear", "polar bears"],
                    'bicycle':["bike", "push-bike", "two-wheeler", "push bicycle", "mountain bike","tandem bicycle"],
                    'bird':["Aves", "fowl", "feathered creature"],
                    'bus':[],
                    'car':[],
                    'cattle':["cow", "oxen", "ox", "bull", "bovine"],
                    'dog':["pup", "puppy", "doggy", "hound", "pooch"],
                    'cat':["kitty","kitten","feline","pussy"],
                    'elephant':["mammoth", "mastodon"],
                    'fox':[],
                    'giant panda':["Ailuropoda melanoleuca","panda bear","panda"],
                    'hamster':[],
                    'horse':["filly","mare","stallion", "bronco", "foal", "mustang","pony"],
                    'lion':["cougar","lioness","puma"],
                    'lizard':["chameleon", "gecko"],
                    'monkey':["ape","primate", "simian","baboon","chimpanzee","gorilla"],
                    'motorcycle':["motorbike"],
                    'rabbit':["bunny","bunny rabbit"],
                    'red panda':["Ailurus fulgens", "lesser panda"],
                    'sheep':["lamb", "ewe"],
                    'snake':["serpent","viper","cobra","python"],
                    'squirrel':[],
                    'tiger':["tigris"],
                    'train':["locomotive","railway train","railway"],
                    'turtle':[],
                    'watercraft':["boat","vessel","cruiser"],
                    'whale':["beluga", "cetacean","orca","finback","grampus","porpoise", "rorqual", "dolphin"],
                    'zebra':[]}

print("Using COCO-stuff as the the background classes")
with open("data/coco_stuff.txt", "r") as f:
    lines = f.readlines()
    lines = [str(x).strip() for x in lines]
    COCO_stuff_class = [x for x in lines if x not in VID_classes]


class Prompts_with_synonym:
    def __init__(self,
                 template_sentences=None,
                 foreground_classes=None,
                 background_classes=None,
                 clip_model=None,
                 device=None):
        if template_sentences is None:
            template_sentences = template_sentence_list
        self.template_sentences = template_sentences

        if foreground_classes is None:
            foreground_classes = VID_classes_dict
        elif isinstance(foreground_classes, list):
            foreground_classes = {i: [] for i in foreground_classes}

        for key in foreground_classes.keys():
            if key not in foreground_classes[key]:
                foreground_classes[key].append(key)
            foreground_classes[key] = list(set(foreground_classes[key]))
        self.foreground_classes = foreground_classes

        if background_classes is None:
            background_classes = dict()
        elif isinstance(background_classes, list):
            background_classes = {i: [] for i in background_classes}

        for key in background_classes.keys():
            if key not in background_classes[key]:
                background_classes[key].append(key)
            background_classes[key] = list(set(background_classes[key]))
        self.background_classes = background_classes

        assert len(set(foreground_classes.keys()).intersection(set(background_classes.keys()))) == 0

        all_classes = {**foreground_classes, **background_classes}
        self.classes = all_classes
      
        self.clip_model = clip_model
        self.device = device
            
        self.prompts = []
        self.prompts_classes_idx = []
        self.prompts_classes = []

        self.classes_name = list(self.classes.keys())
        
        for tmp in template_sentences:
            for cls_idx, cls in enumerate(self.classes.keys()):
                for sub_cls in self.classes[cls]:
                    self.prompts.append(tmp % sub_cls)
                    self.prompts_classes_idx.append(cls_idx)
                    self.prompts_classes.append(cls)  

        self.prompts_classes_idx = np.array(self.prompts_classes_idx)
        self.get_text_features()

    def get_text_features(self):
        self.text_features = get_text_prompts(self.clip_model, self.prompts, self.device)

    def return_foreground_idx(self):
        if hasattr(self, "foreground_idx"):
            return self.foreground_idx
        else:
            self.foreground_idx = (self.prompts_classes_idx < 31) & (self.prompts_classes_idx > 0)
            return self.foreground_idx
        
    def is_foreground_idx(self, prompts_idx):
        classes_idx = self.prompts_classes_idx[prompts_idx]
        return (classes_idx < 31) & (classes_idx > 0)

    def is_same_class(self, prompt_i, prompt_j):
        return self.prompts_classes_idx[prompt_i] == self.prompts_classes_idx[prompt_j]

    def logit_of_classes(self, prompts_logits):
        results_by_classes = {class_idx: [] for class_idx in range(len(self.classes))}
        for class_idx, logit in zip(self.prompts_classes_idx, prompts_logits):
            results_by_classes[class_idx].append(logit.item())
        
        logits_by_classes = []
        for class_idx in range(len(self.classes)):
            logits_by_classes.append(np.max(results_by_classes[class_idx]))
        
        return torch.tensor(logits_by_classes)

    def generate_foreground_prompts(self):
        foreground_prompts = self.__class__(self.template_sentences,
                                            self.foreground_classes,
                                            background_classes=None,
                                            clip_model=self.clip_model,
                                            device=self.device)
        return foreground_prompts
