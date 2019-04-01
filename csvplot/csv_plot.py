import matplotlib
matplotlib.use('Agg')

from flask import Flask, send_from_directory, request, jsonify, render_template
from pathlib import Path
import cgi
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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@click.command()
@click.option('--host', default='127.0.0.1', show_default=True, 
              help="The hostname to listen on. Set this to '0.0.0.0' for the server to be available externally as well")
@click.option('--port', default=8080, show_default=True, help="Port to listen to")
@click.option('--csv', help="CSV file to load")
@click.option('--excel', help="Excel file to load")
@click.option('--delimiter', default=',', show_default=True, help="Delimiter to use in CSV file")
@click.option('--sheet-name', default=0, type=str, help="Excel sheet to load. Defaults to first sheet")
@click.option('--skip-rows', default=0, show_default=True, help="Rows to skip at the beginning")
@click.option('--skip-blank-lines/--include-blank-lines', default=False, show_default=True, help="Skip over blank lines rather than interpreting as NaN values")
def main(host, port, csv, excel, delimiter, sheet_name, skip_rows, skip_blank_lines):
    """Starts the CSV Plot dashboard.
    Loads either a --csv or --excel file for plotting. If neither of these options are given, the built-in Titanic dataset is loaded."""

    if csv is None and excel is None:
        df = sns.load_dataset("titanic")
        name = "Titanic"
    else:
        if csv is not None and excel is not None:
            raise ValueError('Both --csv and --excel flags cannot be set')
        elif csv is not None:
            df = pd.read_csv(
                csv, 
                delimiter=delimiter,
                skiprows=skip_rows, 
                skip_blank_lines=skip_blank_lines,
            )
            name = Path(csv).name
        elif excel is not None:
            df = pd.read_excel(
                excel, 
                sheet_name=sheet_name, 
                skiprows=skip_rows, 
                skip_blank_lines=skip_blank_lines,
            )
            name = Path(excel).name
        else:
            assert False
    
    dashboard(host, port, df, name)


def preprocess_payload(payload):
    form_dict = form_data_to_dict(payload['formData'])

    form_dict = {key.replace('-', '_'): value for key, value in form_dict.items()}

    for key in {'min_x', 'max_x', 'min_y', 'max_y'}:
        form_dict[key] = float_from_string(form_dict[key])
    
    for key in {'facet_width', 'facet_height', 'strip_plot_alpha', 'scatter_alpha'}:
        form_dict[key] = float(form_dict[key])
    
    for key in {'regplot_order'}:
        form_dict[key] = int(form_dict[key])
    
    for key in {'strip_plot_dodge', 'heatmap_square', 'heatmap_annotate', 'box_plot_enhance',
                'violin_plot_split', 'joint_plot_hist', 'joint_plot_kde'}:
        form_dict[key] = key in form_dict
    
    plot_category = form_dict['plot_category']
    plot_group, plot = plot_category.split('--')
    form_dict['plot_group'] = plot_group
    form_dict['plot'] = plot

    form_dict['x_scale'] = 'log' if 'log_x' in form_dict else 'linear'
    form_dict['y_scale'] = 'log' if 'log_y' in form_dict else 'linear'

    field_data = payload['fieldData']

    assert not (set(form_dict) & set(field_data))

    args = AttrDict({**form_dict, **field_data})
    return args

def dashboard(host, port, df, name):
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html', name=name)

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
        return jsonify([str(col) for col in df.columns if dtype_to_type(df[col].dtype) == 'category'])

    @app.route('/getnumberfields', methods=['POST'])
    def getnumberfields():
        return jsonify([str(col) for col in df.columns if dtype_to_type(df[col].dtype) == 'number'])

    @app.route('/getresults', methods=['POST'])
    def getresults():
        try:
            payload = request.json
            args = preprocess_payload(payload)

            aggregation_fn = FUNC_BY_AGGREGATE[args.aggregate]

            aspect_ratio = args.facet_width / args.facet_height

            sns.set_style(args.plot_style)

            if args.min_x is not None and args.max_x is not None and args.min_x >= args.max_x:
                raise ValueError('min_x must be less than max_x')
            if args.min_y is not None and args.max_y is not None and args.min_y >= args.max_y:
                raise ValueError('min_y must be less than max_y')

            if args.plot_group == 'pivot':
                pivot_df = pd.pivot_table(df, values=args.values, index=args.rows, 
                                          columns=args.columns, aggfunc=aggregation_fn)
                
                if isinstance(pivot_df, pd.Series):
                    pivot_df = pivot_df.to_frame().T

                if args.plot == 'pivot-table':
                    table = to_html_table(pivot_df)
                    return table

                elif args.plot == 'heatmap':
                    plt.figure()  # Reset figure
                    fig = sns.heatmap(pivot_df, annot=args.heatmap_annotate, square=args.heatmap_square).get_figure()

                    image = figure_to_pillow_image(fig)
                    base64_image = image_to_base64(image)
                    html = BASE64_HTML_TAG.format(base64_image)

                    return html

                else:
                    raise ValueError('Invalid plot: {}'.format(args.plot))

            if args.plot_group == 'category-plot':
                if len(args.columns) == 0:
                    raise ValueError('At least one column is required')
                elif len(args.columns) > 3:
                    raise ValueError('No more than 3 columns allowed')
                elif len(args.columns) == 1:
                    col, x, hue = None, args.columns[0], None
                elif len(args.columns) == 2:
                    col, x, hue = None, args.columns[0], args.columns[1]
                else:
                    col, x, hue = args.columns

                row = args.rows[0] if args.rows else None

                kwargs = {}
                if args.plot == 'strip':
                    kwargs = {
                        'alpha': args.strip_plot_alpha,
                        'dodge': args.strip_plot_dodge,
                    }
                elif args.plot == 'box':
                    if args.box_plot_enhance:
                        args.plot = 'boxen'
                elif args.plot == 'violin':
                    kwargs = {
                        'inner': args.violin_inner,
                        'split': args.violin_plot_split,
                    }

                images = []
                for y in args.values:
                    g = sns.catplot(x=x, y=y, hue=hue, row=row, col=col, data=df, estimator=aggregation_fn, kind=args.plot, 
                                    margin_titles=True, height=args.facet_height, aspect=aspect_ratio, **kwargs)

                    g.set(ylim=(args.min_y, args.max_y), yscale=args.y_scale)

                    im = figure_to_pillow_image(g)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif args.plot_group == 'relative-plot':
                x = args.xaxis[0]
                color = args.colors[0] if args.colors else None
                shape = args.shapes[0] if args.shapes else None
                size = args.sizes[0] if args.sizes else None
                row = args.rows[0] if args.rows else None
                col = args.columns[0] if args.columns else None

                kwargs = {}
                if args.plot == 'scatter':
                    kwargs = {
                        'alpha': args.scatter_alpha,
                    }

                images = []
                for y in args.values:
                    g = sns.relplot(x=x, y=y, hue=color, size=size, style=shape, row=row, col=col, data=df, kind=args.plot, 
                                    height=args.facet_height, aspect=aspect_ratio, facet_kws={'margin_titles' : True}, **kwargs)

                    g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale, yscale=args.y_scale)
                                      
                    im = figure_to_pillow_image(g)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif args.plot_group == 'regplot':
                x = args.xaxis[0]
                color = args.colors[0] if args.colors else None
                row = args.rows[0] if args.rows else None
                col = args.columns[0] if args.columns else None

                order = args.regplot_order

                images = []
                for y in args.values:
                    g = sns.lmplot(x=x, y=y, hue=color, row=row, col=col, data=df, height=args.facet_height, aspect=aspect_ratio, order=order)
                    g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale, yscale=args.y_scale)
                    im = figure_to_pillow_image(g)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html
            
            elif args.plot_group == 'pair-plot':
                color = args.colors[0] if args.colors else None
                data = df[args.values].fillna(0)  # TODO: drop n/a instead
    
                data = pd.concat((data, df[args.colors]), axis='columns')

                diag_fn_by_key = {
                    'hist': plt.hist, 
                    'kde': sns.kdeplot,
                }

                off_diag_fn_by_key = {
                    'scatter': sns.scatterplot,
                    'kde': sns.kdeplot,
                }

                g = sns.PairGrid(data, hue=color, height=args.facet_height, aspect=aspect_ratio)
                g = g.map_diag(diag_fn_by_key[args.pair_plot_diag])
                g = g.map_upper(off_diag_fn_by_key[args.pair_plot_upper])
                g = g.map_lower(off_diag_fn_by_key[args.pair_plot_lower])
                g = g.add_legend()

                g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale, yscale=args.y_scale)

                im = figure_to_pillow_image(g)
                base64_image = image_to_base64(im)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            elif args.plot_group == 'joint-plot':
                x = args.xaxis[0]

                fn_by_joint_kind = {
                    'scatter': sns.scatterplot,
                    'reg': sns.regplot,
                    'kde': sns.kdeplot,
                }

                images = []
                for y in args.values:
                    # TODO: Min/max x/y + log x/y
                    g = sns.JointGrid(x=x, y=y, data=df, height=args.facet_height)
                    g = g.plot_joint(fn_by_joint_kind[args.joint_plot_kind])
                    g = g.plot_marginals(sns.distplot, hist=args.joint_plot_hist, kde=args.joint_plot_kde)

                    im = figure_to_pillow_image(g)
                    images.append(im)

                image = stack_images(images)
                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)
                return html

            else:
                raise ValueError('Invalid plot_group: {}'.format(args.plot_group))

        except Exception as e:
            traceback.print_exc()
            return cgi.escape(get_class_name(e) + ': ' + str(e)), 400

    app.run(host=host, port=port, debug=True)


def float_from_string(string):
    return float(string) if string else None


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
    form_dict = {key: value if value != 'null' else None for key, value in form_items}

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


def get_class_name(object):
    return object.__class__.__name__


if __name__ == "__main__":
    main()
