
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from nmt import nmt
from nmt import gnmt_model
from nmt import model
from nmt import model_helper
from nmt import attention_model
from nmt.utils import nmt_utils
from nmt.utils import misc_utils as utils
import argparse
import sys
import re
# Windows
import msvcrt
import config as FLAGS

def remove_special_char(s):
    return re.sub("([.,!?\"':;)(])", "", s)

class Chat:
    def __init__(self):
        pass

    def reply(self, s):
        if len(s) == 0 or s == '\r' or s == '\n':
            return ''

        infer_data = [remove_special_char(s)]

        self.sess.run(
            self.infer_model.iterator.initializer,
            feed_dict={
                self.infer_model.src_placeholder: infer_data,
                self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size})

        beam_width = self.hparams.beam_width
        num_translations_per_input = max(
            min(1, beam_width), 1)

        nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
        if beam_width == 0:
            nmt_outputs = np.expand_dims(nmt_outputs, 0)

        batch_size = nmt_outputs.shape[1]

        for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
                translation = nmt_utils.get_translation(
                    nmt_outputs[beam_id],
                    sent_id,
                    tgt_eos=self.hparams.eos,
                    subword_option=self.hparams.subword_option)

        return translation.decode('utf-8')

    def nmt_main(self, flags, default_hparams, scope=None):
        out_dir = flags.out_dir
        if not tf.gfile.Exists(out_dir):
            tf.gfile.MakeDirs(out_dir)

        self.hparams = nmt.create_or_load_hparams(
            out_dir,
            default_hparams,
            flags.hparams_path,
            save_hparams=False
        )

        self.ckpt = tf.train.latest_checkpoint(out_dir)
        if not self.ckpt:
            print('Train is needed')
            sys.exit()

        hparams = self.hparams
        if not hparams.attention:
            model_creator = model.Model
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = gnmt_model.GNMTModel
        else:
            raise ValueError("Unknown model architecture")
        self.infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

        self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())

        with self.infer_model.graph.as_default():
            self.loaded_infer_model = model_helper.load_model(
                self.infer_model.model, self.ckpt, self.sess, 'infer')

    def run(self, flags, default_hparams):
        self.nmt_main(flags, default_hparams)

        # windows에 최적화
        try:
            sys.stdout.write("> ")
            sys.stdout.flush()

            sentence = ''
            c = msvcrt.getwch()
            while True:
                # ctrl + c
                if c == '':
                    self.sess.close()
                    sys.exit()

                sys.stdout.write(c)
                sys.stdout.flush()

                if c == '\r':
                    print ('> {}'.format(sentence))
                    print(self.reply(sentence))
                    sentence = ''
                    sys.stdout.write("> ")
                    sys.stdout.flush()
                else:
                    sentence += c
                c = msvcrt.getwch()
        except KeyboardInterrupt:
            self.sess.close()
            sys.exit()

        '''
        try:
            sys.stdout.write("> ")
            sys.stdout.flush()
            line = sys.stdin.readline()

            while line:
                print ('/{}/'.format(line))
                print(self.reply(line.strip()))
                sys.stdout.write("\n> ")
                sys.stdout.flush()
                line = sys.stdin.readline()
        except KeyboardInterrupt:
            self.sess.close()
            sys.exit()
        '''

def create_hparams(flags):
    return tf.contrib.training.HParams(
        out_dir=flags.out_dir,
        override_loaded_hparams=flags.override_loaded_hparams,
    )

def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    chat = Chat()
    chat.run(FLAGS, default_hparams)

if __name__ == "__main__":
    tf.app.run(main=main, argv=[sys.argv[0]])
