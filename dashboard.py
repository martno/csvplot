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
import traceback
from bs4 import BeautifulSoup
from pprint import pprint
from PIL import Image

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

CATEGORY_PLOTS = {
    'bar',
    'point',
    'strip',
    'box',
    'violin',
}


sns.set()  # Set Seaborn default styles


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

    @app.route('/getcategoryfields', methods=['POST'])
    def getcategoryfields():
        return jsonify(sorted(col for col in df.columns if dtype_to_type(df[col].dtype) == 'category'))

    @app.route('/getnumberfields', methods=['POST'])
    def getnumberfields():
        return jsonify(sorted(col for col in df.columns if dtype_to_type(df[col].dtype) == 'number'))

    @app.route('/getresults', methods=['POST'])
    def getresults():
        try:
            payload = request.json

            print(payload)

            form_dict = form_data_to_dict(payload['formData'])
            
            pprint(form_dict)

            aggregation_fn = FUNC_BY_AGGREGATE[form_dict['aggregate']]

            facet_width = float(form_dict['facet-width'])
            facet_height = float(form_dict['facet-height'])
            aspect_ratio = facet_width / facet_height

            plot_category = form_dict['plot-category']
            plot_group, kind = plot_category.split('--')

            fields = payload['fieldData']
            rows = fields['rows']
            columns = fields['columns']
            values = fields['values']
            colors = fields['colors']
            xaxis = fields['xaxis']
            shapes = fields['shapes']
            sizes = fields['sizes']

            if plot_group == 'pivot':
                pivot_df = pd.pivot_table(df, values=values, index=rows, 
                                          columns=columns, aggfunc=aggregation_fn)
                
                if isinstance(pivot_df, pd.Series):
                    pivot_df = pivot_df.to_frame().T

                if kind == 'pivot-table':
                    table = to_html_table(pivot_df)
                    return table

                elif kind == 'heatmap':
                    plt.figure()  # Reset figure
                    fig = sns.heatmap(pivot_df, annot=True).get_figure()
                    return fig_to_html(fig)

                else:
                    raise ValueError('Invalid kind: {}'.format(kind))

            if plot_group == 'category-plot':
                if len(columns) == 0:
                    raise ValueError('At least one column is required')
                elif len(columns) > 3:
                    raise ValueError('No more than 3 columns allowed')
                elif len(columns) == 1:
                    col, x, hue = None, columns[0], None
                elif len(columns) == 2:
                    col, x, hue = None, columns[0], columns[1]
                else:
                    col, x, hue = columns

                row = rows[0] if rows else None

                images = []
                for y in values:
                    fig = sns.catplot(x=x, y=y, hue=hue, row=row, col=col, data=df, estimator=aggregation_fn, kind=kind, 
                                margin_titles=True, height=facet_height, aspect=aspect_ratio)
                    im = figure_to_pillow_image(fig)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif plot_group == 'relative-plot':
                x = xaxis[0]
                color = colors[0] if colors else None
                shape = shapes[0] if shapes else None
                size = sizes[0] if sizes else None
                row = rows[0] if rows else None
                col = columns[0] if columns else None

                images = []
                for y in values:
                    fig = sns.relplot(x=x, y=y, hue=color, size=size, style=shape, row=row, col=col, data=df, kind=kind, 
                                      height=facet_height, aspect=aspect_ratio)
                    im = figure_to_pillow_image(fig)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif plot_group == 'regplot':
                x = xaxis[0]
                color = colors[0] if colors else None
                row = rows[0] if rows else None
                col = columns[0] if columns else None

                images = []
                for y in values:
                    fig = sns.lmplot(x=x, y=y, hue=color, row=row, col=col, data=df, height=facet_height, aspect=aspect_ratio)
                    im = figure_to_pillow_image(fig)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html
            
            elif plot_group == 'pair-plot':
                color = colors[0] if colors else None

                print(values)

                data = df[values].fillna(0)  # TODO: drop n/a instead
    
                data = pd.concat((data, df[colors]), axis='columns')

                g = sns.PairGrid(data, hue=color, height=facet_height, aspect=aspect_ratio)
                g = g.map_diag(plt.hist)
                g = g.map_upper(sns.kdeplot)
                g = g.map_lower(sns.scatterplot)
                g = g.add_legend()

                im = figure_to_pillow_image(g)
                base64_image = image_to_base64(im)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif plot_group == 'joint-plot':
                x = xaxis[0]

                images = []
                for y in values:
                    g = sns.JointGrid(x=x, y=y, data=df, height=facet_height, aspect=aspect_ratio)
                    g = g.plot_joint(sns.scatterplot)
                    g = g.plot_marginals(sns.distplot)

                    im = figure_to_pillow_image(g)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            else:
                raise ValueError('Invalid plot_group: {}'.format(plot_group))

        except Exception as e:
            traceback.print_exc()
            return cgi.escape(get_class_name(e) + ': ' + str(e)), 400

    app.run(host=host, port=port, debug=True)


def figure_to_pillow_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    im = Image.open(buffer)
    return im


def stack_images(images):
    numpy_images = [np.asarray(image) for image in images]

    new_height = sum(height(image) for image in numpy_images)
    new_width = max(width(image) for image in numpy_images)
    num_channels = numpy_images[0].shape[2]

    new_shape = (new_height, new_width, num_channels)
    stacked_image = np.ones(new_shape, dtype=numpy_images[0].dtype)

    y_offset = 0
    for image in numpy_images:
        stacked_image[y_offset:y_offset+height(image), :width(image), :] = image
        y_offset += height(image)

    return Image.fromarray(stacked_image)


def height(image):
    return image.shape[0]


def width(image):
    return image.shape[1]


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    image_bytes = base64.b64encode(buffered.getvalue())
    image_string = image_bytes.decode('utf-8')

    return image_string


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
