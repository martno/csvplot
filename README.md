# CSV Plot

Make statistical visualizations of CSV and Excel files


## Install

```
pip3 install git+https://github.com/martno/csvplot/
```


## Usage

```
csv-plot --csv myfile.csv
```

Now visit http://localhost:8080/ in your browser to see the dashboard. If the client and the csv-plot server run on different machines, set the flag `--host=0.0.0.0` when starting `csv-plot`. This allows any client with access to the server to see the dashboard.

### See more options

```
csv-plot --help
```


## License

MIT License
