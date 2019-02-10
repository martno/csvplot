import matplotlib
matplotlib.use('Agg')

from flask import Flask, send_from_directory, request, jsonify, render_template
from pathlib import Path
import json
import shutil
import os
import cgi
import re
import click
import io
import base64
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

BASE64_HTML_TAG = '<img src="data:image/png;base64,{}">'

FUNC_BY_AGGREGATE = {
    'average': np.mean,
    'sum': np.sum,
}


@click.command()
@click.option('--host', default='127.0.0.1', show_default=True, 
              help="The hostname to listen on. Set this to '0.0.0.0' to have the server available externally as well")
@click.option('--port', default=8080, show_default=True, help="Port to listen to")
def main(host, port):
    df = sns.load_dataset("titanic")
    dashboard(host, port, df)


def dashboard(host=None, port=None, df=None):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory('static', 'favicon.ico')

    @app.route('/js/<path:path>')
    def send_js(path):
        return send_from_directory('js', path)

    @app.route('/css/<path:path>')
    def send_css(path):
        return send_from_directory('css', path)

    @app.route('/getfields', methods=['POST'])
    def getfields():
        return jsonify({col: dtype_to_type(df[col].dtype) for col in df.columns})

    @app.route('/getresults', methods=['POST'])
    def getresults():
        try:
            payload = request.json

            form_dict = form_data_to_dict(payload['formData'])
            aggregation_fn  = FUNC_BY_AGGREGATE[form_dict['aggregate']]
            kind = form_dict['kind']

            fields = payload['fieldData']
            rows = fields['rows']
            columns = fields['columns']
            values = fields['values']

            if kind == 'pivot-table':
                pivot_df = pd.pivot_table(df, values=values, index=rows, 
                                        columns=columns, aggfunc=aggregation_fn)
                table = to_html_table(pivot_df)
                return table

            elif kind == 'barplot':
                if len(columns) == 0:
                    raise ValueError('At least one column is required')
                elif len(columns) == 1:
                    col, x, hue = None, columns[0], None
                elif len(columns) == 2:
                    col, x, hue = None, columns[0], columns[1]
                else:
                    col, x, hue = columns[-3:]

                row = rows[-1] if rows else None

                y = values[0]
                
                g = sns.catplot(x=x, y=y, hue=hue, row=row, col=col, data=df, estimator=aggregation_fn, kind='bar', 
                            margin_titles=True)
                return fig_to_html(g)

            else:
                raise ValueError('Invalid kind: {}'.format(kind))

        except Exception as e:
            raise
            return cgi.escape(get_class_name(e) + ': ' + str(e)), 400

    app.run(host=host, port=port, debug=True)


def dtype_to_type(dtype):
    if dtype.name in ('category', 'object', 'bool'):
        return 'category'
    return 'number'


def form_data_to_dict(form_data):
    form_items = form_data.split('&')
    form_items = [item.split('=') for item in form_items]
    form_dict = {key: value for key, value in form_items}

    return form_dict


def to_html_table(df):
    cm = sns.light_palette("green", as_cmap=True)
    html = df.style.apply(
        table_background_gradient,
        cmap=cm,
        m=df.min().min(),
        M=df.max().max(),
        high=0.2,
    )

    html.format('{:.2f}')  # Round all floats to 2 decimals

    html = html.render()

    soup = BeautifulSoup(html, 'html.parser')
    soup.table['class'] = 'table table-sm'
    soup.table['style'] = 'width: auto;'

    return soup.prettify()


def table_background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    """The background gradient is per table, instead of either per row or per column"""
    # https://stackoverflow.com/questions/38931566/pandas-style-background-gradient-both-rows-and-columns
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def fig_to_html(g):
    image_string = convert_fig_to_base64(g)
    html = BASE64_HTML_TAG.format(image_string)
    return html


def convert_fig_to_base64(g):
    buffered = io.BytesIO()
    g.savefig(buffered, format='png')
    image_bytes = base64.b64encode(buffered.getvalue())
    image_string = image_bytes.decode('utf-8')

    return image_string


def get_class_name(object):
    return object.__class__.__name__


if __name__ == "__main__":
    main()
