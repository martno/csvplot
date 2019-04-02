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
    'swarm',
    'box',
    'violin',
}


sns.set()  # Set Seaborn default styles


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='127.0.0.1', show_default=True,
              help="The hostname to listen on. Set this to '0.0.0.0' for the server to be available externally as well")
@click.option('--port', default=8080, show_default=True, help="Port to listen to")
@click.option('--csv', help="CSV file to load")
@click.option('--excel', help="Excel file to load")
@click.option('--delimiter', default=None, show_default=True, help="Delimiter to use in CSV file")
@click.option('--sheet-name', default=None, type=str, help="Excel sheet to load. Defaults to first sheet")
def dashboard(host, port, csv, excel, delimiter, sheet_name):
    """Starts the CSV Plot dashboard.
    Loads either a --csv or --excel file for plotting. If neither of these options are given, the built-in Titanic dataset is loaded."""

    df, name = load_data(csv, delimiter, excel, sheet_name)

    kwargs = {
        'csv': csv,
        'excel': excel,
        'delimiter': delimiter,
        'sheet_name': sheet_name,
    }

    dashboard(host, port, df, name, kwargs)


def load_data(csv, delimiter, excel, sheet_name):
    if delimiter is None:
        delimiter = ','
    if sheet_name is None:
        sheet_name = 0

    if csv is not None and excel is not None:
        raise ValueError('Both --csv and --excel flags cannot be set')
    elif csv is not None:
        df = pd.read_csv(
            csv,
            delimiter=delimiter,
        )
        name = Path(csv).name
    elif excel is not None:
        df = pd.read_excel(
            excel,
            sheet_name=sheet_name,
        )
        name = Path(excel).name
    else:
        assert csv is None and excel is None
        df = sns.load_dataset("titanic")
        name = "Titanic"

    return df, name


def dashboard(host, port, df, name, kwargs):
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

            for key, value in kwargs.items():
                args[key] = value

            sns.set_style(args.plot_style)

            if args.min_x is not None and args.max_x is not None and args.min_x >= args.max_x:
                raise ValueError('min_x must be less than max_x')
            if args.min_y is not None and args.max_y is not None and args.min_y >= args.max_y:
                raise ValueError('min_y must be less than max_y')

            if args.plot_group == 'pivot' and args.plot == 'pivot-table':
                pivot_df = to_pivot_df(df, args)
                table = to_html_table(pivot_df)
                return table

            else:
                plot_fn = PLOT_FN_BY_NAME[args.plot_group]
                image = plot_fn(df, args)

                base64_image = image_to_base64(image)
                html = BASE64_HTML_TAG.format(base64_image)

                flags = DEFAULT_FLAGS | FLAGS_BY_PLOT_NAME[args.plot_group]
                script = get_script(args, flags)

                return create_plot_tabs(html, script)

        except Exception as e:
            traceback.print_exc()
            return cgi.escape(get_class_name(e) + ': ' + str(e)), 400

    app.run(host=host, port=port, debug=True)


def dtype_to_type(dtype):
    if dtype.name in ('category', 'object', 'bool'):
        return 'category'
    return 'number'


def preprocess_payload(payload):
    form_dict = form_data_to_dict(payload['formData'])

    form_dict = {key.replace('-', '_'): value for key, value in form_dict.items()}

    for key in {'min_x', 'max_x', 'min_y', 'max_y'}:
        form_dict[key] = float_from_string(form_dict[key])

    for key in {'facet_width', 'facet_height', 'strip_plot_alpha', 'scatter_alpha'}:
        form_dict[key] = float(form_dict[key])

    for key in {'regplot_order', 'rotate_x_label'}:
        form_dict[key] = int(form_dict[key])

    checkboxes = {'strip_plot_dodge', 'heatmap_square', 'heatmap_annotate', 'box_plot_enhance',
                  'violin_plot_split', 'joint_plot_hist', 'joint_plot_kde', 'bar_plot_annotate'}
    for key in checkboxes:
        form_dict[key] = key in form_dict

    plot_category = form_dict['plot_category']
    plot_group, plot = plot_category.split('--')
    form_dict['plot_group'] = plot_group
    form_dict['plot'] = plot

    form_dict['x_scale'] = 'log' if 'log_x' in form_dict else 'linear'
    form_dict['y_scale'] = 'log' if 'log_y' in form_dict else 'linear'

    form_dict['aspect_ratio'] = form_dict['facet_width'] / form_dict['facet_height']

    field_data = payload['fieldData']

    assert not (set(form_dict) & set(field_data))

    args = AttrDict({**form_dict, **field_data})
    return args


def form_data_to_dict(form_data):
    form_items = form_data.split('&')
    form_items = [item.split('=') for item in form_items]
    form_dict = {key: (value if value != 'null' else None) for key, value in form_items}

    return form_dict


def float_from_string(string):
    return float(string) if string else None


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


def create_heatmap(df, args):
    pivot_df = to_pivot_df(df, args)
    image = create_heatmap_from_pivot_df(pivot_df, args)
    return image


def to_pivot_df(df, args):
    aggregation_fn = FUNC_BY_AGGREGATE[args.aggregate]
    pivot_df = pd.pivot_table(df, values=args.values, index=args.rows,
                              columns=args.columns, aggfunc=aggregation_fn)

    if isinstance(pivot_df, pd.Series):
        pivot_df = pivot_df.to_frame().T

    return pivot_df


def create_heatmap_from_pivot_df(pivot_df, args):
    plt.figure()  # Reset figure

    fig = sns.heatmap(pivot_df, annot=args.heatmap_annotate, square=args.heatmap_square).get_figure()
    image = figure_to_pillow_image(fig)
    return image


def create_category_plot(df, args):
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

    aggregation_fn = FUNC_BY_AGGREGATE[args.aggregate]

    images = []
    for y in args.values:
        g = sns.catplot(x=x, y=y, hue=hue, row=row, col=col, data=df, estimator=aggregation_fn, kind=args.plot,
                        margin_titles=True, height=args.facet_height, aspect=args.aspect_ratio, **kwargs)

        g.set(ylim=(args.min_y, args.max_y), yscale=args.y_scale)
        g.set_xticklabels(rotation=args.rotate_x_label)

        if args.plot == 'bar' and args.bar_plot_annotate:
            autolabel_ax(g.axes)

        im = figure_to_pillow_image(g)
        images.append(im)

    image = stack_images(images)
    return image


def autolabel_ax(ax):
    if type(ax) == np.ndarray:
        for a in ax:
            autolabel_ax(a)
        return

    for c in ax.containers:
        rects = c.get_children()
        autolabel(ax, rects, xpos='left')


def autolabel(ax, rects, xpos):
    """
    From https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '%.2f' % height, ha=ha[xpos], va='bottom')


def create_relative_plot(df, args):
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
                        height=args.facet_height, aspect=args.aspect_ratio, facet_kws={'margin_titles': True}, **kwargs)

        g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale,
              yscale=args.y_scale)

        im = figure_to_pillow_image(g)
        images.append(im)

    image = stack_images(images)
    return image


def create_regplot(df, args):
    x = args.xaxis[0]
    color = args.colors[0] if args.colors else None
    row = args.rows[0] if args.rows else None
    col = args.columns[0] if args.columns else None
    order = args.regplot_order

    images = []
    for y in args.values:
        g = sns.lmplot(x=x, y=y, hue=color, row=row, col=col, data=df, height=args.facet_height,
                       aspect=args.aspect_ratio, order=order)
        g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale,
              yscale=args.y_scale)

        im = figure_to_pillow_image(g)
        images.append(im)
    image = stack_images(images)

    return image


def create_pair_plot(df, args):
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

    g = sns.PairGrid(data, hue=color, height=args.facet_height, aspect=args.aspect_ratio)

    g = g.map_diag(diag_fn_by_key[args.pair_plot_diag])
    g = g.map_upper(off_diag_fn_by_key[args.pair_plot_upper])
    g = g.map_lower(off_diag_fn_by_key[args.pair_plot_lower])

    g = g.add_legend()
    g.set(xlim=(args.min_x, args.max_x), ylim=(args.min_y, args.max_y), xscale=args.x_scale, yscale=args.y_scale)

    im = figure_to_pillow_image(g)
    return im


def create_joint_plot(df, args):
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

    return image


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


def get_script(args, flags):
    on_off_flags_by_flag = {param.name: {True: param.opts[0], False: param.secondary_opts[0]}
                            for param in generate.params if param.is_bool_flag}

    boolean_flags = []
    for flag in flags:
        if flag in on_off_flags_by_flag:
            boolean_flags.append(on_off_flags_by_flag[flag][args[flag]])

    value_by_flag = {flag: args[flag] for flag in flags if flag not in on_off_flags_by_flag}
    value_by_flag = {flag.replace('_', '-'): process_arg(value) for flag, value in value_by_flag.items()}
    value_by_flag = {flag: value for flag, value in value_by_flag.items() if value is not None and value != ''}

    sorted_flags = sorted(value_by_flag.keys())
    script = 'csv-plot generate ' \
           + ' '.join('--{} {}'.format(flag, value_by_flag[flag]) for flag in sorted_flags) + ' ' \
           + ' '.join(boolean_flags) \
           + ' --output output.png'

    script = script.replace('--', '\\\n    --')

    return cgi.escape(script)


def process_arg(arg):
    if type(arg) == list:
        return ','.join(arg)
    return arg


def create_plot_tabs(plot, script):
    return '''
    <div>
      <ul class="nav nav-pills" id="sidebar-tabs" role="tablist">
        <li class="nav-item">
          <a data-toggle="pill" class="nav-link active text-center" href="#plot-tab" role="tab">Plot</a>
        </li>
        <li class="nav-item">
          <a data-toggle="pill" class="nav-link text-center" href="#regenerate-tab" role="tab">Regenerate</a>
        </li>
      </ul>

      <div class="tab-content">
        <div id="plot-tab" role="tabpanel" class="tab-pane active">
          {plot}
        </div>
        <div id="regenerate-tab" role="tabpanel" class="tab-pane">
          <textarea class="form-control" id="exampleFormControlTextarea1" rows="{rows}" style="width: 800px;">{script}</textarea>
        </div>
      </div>
    </div>
    '''.format(plot=plot, rows=len(script.split('\n')), script=script)


def get_class_name(object):
    return object.__class__.__name__


PLOT_GROUP_BY_PLOT = {
    'heatmap': 'pivot',
    'bar': 'category-plot',
    'strip': 'category-plot',
    'swarm': 'category-plot',
    'violin': 'category-plot',
    'point': 'category-plot',
    'scatter': 'relative-plot',
    'line': 'relative-plot',
    'regplot': 'regplot',
    'pair-plot': 'pair-plot',
    'joint-plot': 'joint-plot',
}


@cli.command()
@click.option('--csv', help="CSV file to load")
@click.option('--excel', help="Excel file to load")
@click.option('--delimiter', default=',', show_default=True, help="Delimiter to use in CSV file")
@click.option('--sheet-name', default=0, type=str, help="Excel sheet to load. Defaults to first sheet")
@click.option('--output', help="Output filename", required=True)
@click.option('--plot', help="Plot type", required=True,
              type=click.Choice(
                  ['heatmap', 'bar', 'strip', 'swarm', 'box', 'violin', 'point', 'scatter', 'line', 'regplot', 'pair-plot',
                   'joint-plot']))
@click.option('--rows', default='', help="Row fields (separate by comma)")
@click.option('--columns', default='', help="Column fields (separate by comma)")
@click.option('--values', default='', help="Value fields (separate by comma)")
@click.option('--colors', default='', help="Color field")
@click.option('--xaxis', default='', help="X axis field")
@click.option('--shapes', default='', help="Shape field")
@click.option('--sizes', default='', help="Size field")
@click.option('--aggregate', default='average', show_default=True, help="Aggregation function", type=click.Choice(['average', 'sum']))
@click.option('--facet-width', default=4.0, show_default=True, help="Facet width")
@click.option('--facet-height', default=4.0, show_default=True, help="Facet height")
@click.option('--plot-style', default='darkgrid', show_default=True, help="Plot style",
              type=click.Choice(['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']))
@click.option('--min-x', default=None, type=float, help="Min x")
@click.option('--max-x', default=None, type=float, help="Max x")
@click.option('--min-y', default=None, type=float, help="Min y")
@click.option('--max-y', default=None, type=float, help="Max y")
@click.option('--x-scale', default='linear', show_default=True, type=click.Choice(['linear', 'log']))
@click.option('--y-scale', default='linear', show_default=True, type=click.Choice(['linear', 'log']))
@click.option('--rotate-x-label', default=0, help="Rotate x label")
@click.option('--heatmap-square/--no-heatmap-square', default=False, show_default=True)
@click.option('--heatmap-annotate/--no-heatmap-annotate', default=False, show_default=True)
@click.option('--bar-plot-annotate/--bar-plot-no-annotate', default=False, show_default=True)
@click.option('--strip-plot-alpha', default=1.0, show_default=True)
@click.option('--strip-plot-dodge/--strip-plot-no-dodge', default=False, show_default=True)
@click.option('--box-plot-enhance/--box-plot-no-enhance', default=False, show_default=True)
@click.option('--violin-inner', default='box', show_default=True,
              type=click.Choice(['box', 'quartile', 'point', 'stick', 'None']))
@click.option('--violin-plot-split/--violin-plot-no-split', default=False, show_default=True)
@click.option('--scatter-alpha', default=1.0, show_default=True)
@click.option('--regplot-order', default=1, show_default=True)
@click.option('--pair-plot-diag', default='hist', show_default=True, type=click.Choice(['hist', 'kde']))
@click.option('--pair-plot-upper', default='scatter', show_default=True, type=click.Choice(['scatter', 'kde']))
@click.option('--pair-plot-lower', default='scatter', show_default=True, type=click.Choice(['scatter', 'kde']))
@click.option('--joint-plot-kind', default='scatter', show_default=True, type=click.Choice(['scatter', 'reg', 'kde']))
@click.option('--joint-plot-hist', default=True, show_default=True)
@click.option('--joint-plot-kde', default=True, show_default=True)
def generate(**kwargs):
    """Generates plots."""
    kwargs = {key: (value if value != 'None' else None) for key, value in kwargs.items()}

    for key in {'rows', 'columns', 'values', 'colors', 'xaxis', 'shapes', 'sizes'}:
        kwargs[key] = [v for v in kwargs[key].split(',') if v]

    kwargs['plot_group'] = PLOT_GROUP_BY_PLOT[kwargs['plot']]

    kwargs['aspect_ratio'] = kwargs['facet_width'] / kwargs['facet_height']

    args = AttrDict(kwargs)

    df, name = load_data(args.csv, args.delimiter, args.excel, args.sheet_name)

    sns.set_style(args.plot_style)

    if args.min_x is not None and args.max_x is not None and args.min_x >= args.max_x:
        raise ValueError('min_x must be less than max_x')
    if args.min_y is not None and args.max_y is not None and args.min_y >= args.max_y:
        raise ValueError('min_y must be less than max_y')

    plot_fn = PLOT_FN_BY_NAME[args.plot_group]
    image = plot_fn(df, args)

    image.save(args.output)


PLOT_FN_BY_NAME = {
    'pivot': create_heatmap,
    'category-plot': create_category_plot,
    'relative-plot': create_relative_plot,
    'regplot': create_regplot,
    'pair-plot': create_pair_plot,
    'joint-plot': create_joint_plot,
}


DEFAULT_FLAGS = {
    'plot',
    'csv',
    'excel',
    'delimiter',
    'sheet_name',
}


FLAGS_BY_PLOT_NAME = {
    'pivot': {
        'aggregate',
        'values',
        'rows',
        'columns',
        'heatmap_annotate',
        'heatmap_square',
    },
    'category-plot': {
        'columns',
        'rows',
        'strip_plot_alpha',
        'strip_plot_dodge',
        'box_plot_enhance',
        'violin_inner',
        'violin_plot_split',
        'aggregate',
        'values',
        'facet_height',
        'facet_width',
        'min_y',
        'max_y',
        'y_scale',
        'rotate_x_label',
        'bar_plot_annotate',
    },
    'relative-plot': {
        'xaxis',
        'values',
        'colors',
        'shapes',
        'sizes',
        'rows',
        'columns',
        'scatter_alpha',
        'facet_height',
        'facet_width',
        'min_x',
        'max_x',
        'min_y',
        'max_y',
        'x_scale',
        'y_scale',
    },
    'regplot': {
        'xaxis',
        'values',
        'colors',
        'rows',
        'columns',
        'regplot_order',
        'facet_height',
        'facet_width',
        'min_x',
        'max_x',
        'min_y',
        'max_y',
        'x_scale',
        'y_scale',
    },
    'pair-plot': {
        'colors',
        'values',
        'facet_height',
        'facet_width',
        'pair_plot_diag',
        'pair_plot_upper',
        'pair_plot_lower',
        'min_x',
        'max_x',
        'min_y',
        'max_y',
        'x_scale',
        'y_scale',
    },
    'joint-plot': {
        'xaxis',
        'values',
        'facet_height',
        'joint_plot_kind',
        'joint_plot_hist',
        'joint_plot_kde',
    },
}


if __name__ == "__main__":
    cli()
