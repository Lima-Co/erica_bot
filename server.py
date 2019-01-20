
# -*- coding: UTF-8 -*-

import os
import re
import math
import time
import json
import logging
import datetime
from logging.handlers import RotatingFileHandler
from flask import Flask, request, session, g, redirect, url_for, \
    abort, render_template, flash, make_response, jsonify, json, \
    send_from_directory
from flask.views import MethodView
from uadetector.flask import UADetector
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import quote, unquote
import chat

application = Flask(__name__)
UADetector(application)

chat.FLAGS.out_dir = 'nmt_model/'
chatbot = chat.Chat()
default_hparams = chat.create_hparams(chat.FLAGS)
chatbot.nmt_main(chat.FLAGS, default_hparams)

@application.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               '/images/favicon.ico', mimetype='image/vnd.microsoft.icon')

@application.route('/')
def root():
    return render_template('home.html')

@application.route('/reply/<message>/', methods=['GET'])
def reply(message):
    if request.method == 'GET':
        answer = chatbot.reply(message)
        print(answer)
        response = {
            'answer': answer
        }
        return json.dumps(response)

@application.route('/<path:path>', methods=['GET'])
def router(path):
    print (path)
    return render_template(path)

if __name__ == '__main__':
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=2)
    handler.setFormatter(logging.Formatter(
        '[%(levelname)s:%(name)s: %(message)s in %(asctime)s; %(filename)s:%(lineno)d'
    ))
    handler.setLevel(logging.DEBUG)
    #application.debug = False
    application.logger.addHandler(handler)
    application.run(host='0.0.0.0')
