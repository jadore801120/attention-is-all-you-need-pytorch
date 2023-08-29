import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs

from .. import data


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)


    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)



class Multi30k(TranslationDataset):
    """The small-dataset WMT 2016 multimodal task, also known as Flickr30k"""

    urls = ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/'
            'wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    name = 'multi30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test='test2016', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Multi30k, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)



class IWSLT(TranslationDataset):
    """The IWSLT 2016 TED talk translation task"""

    base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    name = 'iwslt'
    base_dirname = '{}-{}'

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='IWSLT16.TED.tst2013',
               test='IWSLT16.TED.tst2014', **kwargs):
        """Create dataset objects for splits of the IWSLT dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        cls.dirname = cls.base_dirname.format(exts[0][1:], exts[1][1:])
        cls.urls = [cls.base_url.format(exts[0][1:], exts[1][1:], cls.dirname)]
        check = os.path.join(root, cls.name, cls.dirname)
        path = cls.download(root, check=check)

        train = '.'.join([train, cls.dirname])
        validation = '.'.join([validation, cls.dirname])
        if test is not None:
            test = '.'.join([test, cls.dirname])

        if not os.path.exists(os.path.join(path, train) + exts[0]):
            cls.clean(path)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


    @staticmethod
    def clean(path):
        for f_xml in glob.iglob(os.path.join(path, '*.xml')):
            print(f_xml)
            f_txt = os.path.splitext(f_xml)[0]
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
                root = ET.parse(f_xml).getroot()[0]
                for doc in root.findall('doc'):
                    for e in doc.findall('seg'):
                        fd_txt.write(e.text.strip() + '\n')

        xml_tags = ['<url', '<keywords', '<talkid', '<description',
                    '<reviewer', '<translator', '<title', '<speaker']
        for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
            print(f_orig)
            f_txt = f_orig.replace('.tags', '')
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
                    io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
                for l in fd_orig:
                    if not any(tag in l for tag in xml_tags):
                        fd_txt.write(l.strip() + '\n')



class WMT14(TranslationDataset):
    """The WMT 2014 English-German dataset, as preprocessed by Google Brain.

    Though this download contains test sets from 2015 and 2016, the train set
    differs slightly from WMT 2015 and 2016 and significantly from WMT 2017."""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=0B_bZck-ksdkpM25jRUN2X2UxMm8', 'wmt16_en_de.tar.gz')]
    name = 'wmt14'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train.tok.clean.bpe.32000',
               validation='newstest2013.tok.bpe.32000',
               test='newstest2014.tok.bpe.32000', **kwargs):
        """Create dataset objects for splits of the WMT 2014 dataset.

        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.en', '.de') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default:
                'train.tok.clean.bpe.32000'.
            validation: The prefix of the validation data. Default:
                'newstest2013.tok.bpe.32000'.
            test: The prefix of the test data. Default:
                'newstest2014.tok.bpe.32000'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(WMT14, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)
