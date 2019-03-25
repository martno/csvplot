# CSV Plot

Make statistical visualizations of CSV and Excel files in a dashboard


![Dashboard](https://user-images.githubusercontent.com/176676/54926047-e1758a00-4f52-11e9-934b-b92f418d3c84.gif)


## Install

```
pip3 install git+https://github.com/martno/csvplot/
```


## Usage

```
csv-plot --csv myfile.csv
```

Now visit http://localhost:8080/ in your browser to see the dashboard. If the client and the csv-plot server run on different machines, set the flag `--host=0.0.0.0` when starting `csv-plot`. This allows any client with access to the server to see the dashboard.

### Load built-in Titanic dataset

```
csv-plot
```

### See more options

```
csv-plot --help
```


## License

MIT License
