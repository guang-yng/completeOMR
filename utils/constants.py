from mung.io import parse_node_classes

__all__ = ["CLASS_DICT_ALL", "CLASS_LIST_20", "CLASS_LIST_ESSN", "get_classlist_and_classdict"]
# __all__ = ["node_classes_dict", "node_class_dict_count100", "RESTRICTEDCLASSES20", "ESSENTIALCLASSES"]

node_classes_path = "data/MUSCIMA++/v2.0/specifications/mff-muscima-mlclasses-annot.xml"
# node_classes_path = "resources/mff-muscima-mlclasses-annot.xml"
# filepath = "resources/mff-muscima-mlclasses-annot.deprules"

# class_distribution = "Yolo Result Analysis - Train Res.csv"

CLASS_DICT_ALL = {node_class.name : node_class.class_id for node_class in parse_node_classes(node_classes_path)}

# with open("utils/train_class_dist.json", 'r') as f:
#     cls_count = json.load(f)
# node_class_dict_count100 = {name : id for name, id in node_classes_dict.items() 
#                             if name in cls_count and cls_count[name] >= 100}

# 20 Restricted classes
CLASS_LIST_20 = [
    "noteheadHalf", "noteheadWhole", 
    "noteheadFull",
    "stem",
    "beam",
    "legerLine",
    "augmentationDot",
    "slur",
    "rest8th",
    "accidentalNatural",
    "accidentalFlat",
    "accidentalSharp",
    "barline" 
    "gClef",
    "fClef",
    "dynamicLetterP",
    "dynamicLetterM",
    "dynamicLetterF",
    "keySignature",
    "flag8thUp",
    "flag8thDown",
]

# Essential classes
CLASS_LIST_ESSN = [
    'noteheadFull' 
    'stem' 
    'beam' 
    'augmentationDot' 
    'accidentalSharp' 
    'accidentalFlat' 
    'accidentalNatural' 
    'accidentalDoubleSharp' 
    'accidentalDoubleFlat',
	'restWhole',
	'restHalf',
	'restQuarter',
	'rest8th',
	'rest16th',
	'multiMeasureRest',
	'repeat1Bar',
	'legerLine',
	'graceNoteAcciaccatura',
	'noteheadFullSmall',
	'brace',
	'staffGrouping',
	'barline',
	'barlineHeavy',
	'measureSeparator',
	'repeat',
	'repeatDot',
	'articulationStaccato',
	'articulationTenuto',
	'articulationAccent',
	'slur',
	'tie',
	'dynamicCrescendoHairpin',
	'dynamicDiminuendoHairpin',
	'ornament',
	'wiggleTrill',
	'ornamentTrill',
	'arpeggio',
	'glissando',
	'tupleBracket',
	'tuple',
	'gClef',
	'fClef',
	'cClef',
	'keySignature',
	'timeSignature',
	'dynamicsText',
	'tempoText',
	'otherText',
	'numeral0',
	'numeral1',
	'numeral2',
	'numeral3',
	'numeral4',
	'numeral5',
	'numeral6',
	'numeral7',
	'numeral8',
	'otherNumericSign',
	'unclassified',
	'horizontalSpanner',
	'breathMark',
	'noteheadHalf',
	'noteheadWhole',
	'flag8thUp',
	'flag8thDown',
	'flag16thUp',
	'flag16thDown',
	'fermataAbove',
	'fermataBelow',
	'dynamicLetterP',
	'dynamicLetterM',
	'dynamicLetterF',
	'dynamicLetterS'
]

def get_classlist_and_classdict(class_schema):
    assert class_schema in ["20", "essential", "essn", "all"], f"Invalid class schema {class_schema}"
    class_dict = CLASS_DICT_ALL
    if class_schema == '20':
        class_list = CLASS_LIST_20
        class_dict["noteheadWhole"] = class_dict["noteheadHalf"]
    elif class_schema in ['essential', 'essn']:
        class_list = CLASS_LIST_ESSN
    else:
        class_list = class_dict.keys()
    return class_list, class_dict