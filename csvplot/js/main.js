"use strict";


var LIMITS_BY_CATEGORY = {
    'pivot': {
        'rows': {'min': 0, 'max': Infinity},
        'columns': {'min': 0, 'max': Infinity},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 0},
        'xaxis': {'min': 0, 'max': 0},
        'shapes': {'min': 0, 'max': 0},
        'sizes': {'min': 0, 'max': 0}
    },
    'category-plot': {
        'rows': {'min': 0, 'max': 1},
        'columns': {'min': 1, 'max': 3},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 0},
        'xaxis': {'min': 0, 'max': 0},
        'shapes': {'min': 0, 'max': 0},
        'sizes': {'min': 0, 'max': 0}
    },
    'relative-plot': {
        'rows': {'min': 0, 'max': 1},
        'columns': {'min': 0, 'max': 1},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 1},
        'xaxis': {'min': 1, 'max': 1},
        'shapes': {'min': 0, 'max': 1},
        'sizes': {'min': 0, 'max': 1}
    },
    'regplot': {
        'rows': {'min': 0, 'max': 1},
        'columns': {'min': 0, 'max': 1},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 1},
        'xaxis': {'min': 1, 'max': 1},
        'shapes': {'min': 0, 'max': 0},
        'sizes': {'min': 0, 'max': 0}
    },
    'pair-plot': {
        'rows': {'min': 0, 'max': 0},
        'columns': {'min': 0, 'max': 0},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 1},
        'xaxis': {'min': 1, 'max': 0},
        'shapes': {'min': 0, 'max': 0},
        'sizes': {'min': 0, 'max': 0}
    },
    'joint-plot': {
        'rows': {'min': 0, 'max': 0},
        'columns': {'min': 0, 'max': 0},
        'values': {'min': 1, 'max': Infinity},
        'colors': {'min': 0, 'max': 0},
        'xaxis': {'min': 1, 'max': 1},
        'shapes': {'min': 0, 'max': 0},
        'sizes': {'min': 0, 'max': 0}
    }
};

var FIELD_NAMES = [
    'rows',
    'columns',
    'values',
    'colors',
    'xaxis',
    'shapes',
    'sizes'
]


$(document).ready(function() {
    var drake = dragula(
        $('.collection').toArray(), 
        {
            moves: function (element, source, handle, sibling) {
                return $(element).hasClass('field');
            },
            accepts: function(element, target, source, sibling) {
                for (let kind of ['number', 'category']) {
                    if ($(element).hasClass(kind) && $(target).hasClass(kind)) {
                        return true;
                    }
                }
                return false;
            }
        }
    );

    drake.on('drop', function(element, target, source, sibling) {
        updateFields();
    });

    var promise = postJson('/getcategoryfields');
    promise.done(function(fields) {
        for (let field of fields) {
            $('#field-categories').append(`<div class="category field">${field}</div>`);
        }
    });

    var promise = postJson('/getnumberfields');
    promise.done(function(fields) {
        for (let field of fields) {
            $('#field-numbers').append(`<div class="number field">${field}</div>`);
        }
    });

    $('#submit').click(function() {
        var fieldData = {};
        for (let variable of FIELD_NAMES) {
            var fields = [];
            var field_divs = $('#field-' + variable).children('div').toArray();
            for (let field_div of field_divs) {
                if (!$(field_div).hasClass('transparent')) {
                    fields.push($(field_div).text());
                }                
            }
            fieldData[variable] = fields;
        }

        var json_data = {
            'fieldData': fieldData,
            'formData': $('#field-form').serialize()
        };

        $('#submit').attr('disabled', true);

        var promise = postJson('/getresults', json_data);
        promise.done(function(content) {
            $('#main-window').html(content);
            $('#submit').attr('disabled', false);
        });
        promise.fail(function(content) {
            alert('[Error] ' + content.responseText);
            $('#submit').attr('disabled', false);
        });
    });

    $('input[type=radio][name=plot-category]').change(function() {
        updateFields();
        updatePlotOptions();
    });

    updateFields();
    updatePlotOptions();

    enableTooltips();
});


function updateFields() {
    $('.collection').find('.field').removeClass('transparent');

    for (let container of FIELD_NAMES) {
        var fields = $(`#field-${container}`).find('.field').toArray();

        var plotCategory = getSelectedRadioButton('#field-form', 'plot-category');
        var category = plotCategory.split('--')[0];

        var rangeByContainer = LIMITS_BY_CATEGORY[category];
        var range = rangeByContainer[container]
        var numOpaque = range.max;

        updateTransparentFields(fields, numOpaque);

        var rangeString = rangeToString(range);
        $(`#${container}-display`).html(rangeString);
    }
}


function getSelectedRadioButton(formId, groupName) { 
    return $(`input[name=${groupName}]:checked`, formId).val();
}


function updateTransparentFields(fields, numOpaque) {
    for (let i of range(fields.length)) {
        var field = $(fields[i]);
        if (i < fields.length - numOpaque) {
            field.addClass('transparent');
        } else {
            field.removeClass('transparent');                
        }
    };
}


function range(n) {
    return Array(n).keys();
}


function rangeToString(range) {
    var min = range.min;
    var max = range.max;
    if (max === Infinity) {
        max = '&infin;';
    }

    if (max === 0) {
        return 'Disabled';
    }

    if (min === max) {
        return `${min}`;
    }
    return `${min}&rarr;${max}`
}


function updatePlotOptions() {
    $('.plot-option').hide();

    var plotCategory = getSelectedRadioButton('#field-form', 'plot-category');
    $('#' + plotCategory).show();
}


function enableTooltips() {
    $('[data-toggle="tooltip"]').tooltip();
}


function postJson(url, data) {
    var promise = $.ajax({
        type: "POST",
        url: url,
        data: JSON.stringify(data),
        contentType: 'application/json; charset=utf-8',
    });

    return promise;
}


function deleteRequest(url) {
    return $.ajax({
        url: url,
        type: 'DELETE',
    });
}


function hasMethod(obj, method) {
    for (var id in obj) {
        if (typeof(obj[id]) == "function" && id === method) {
            return true;
        }
    }
    return false;
}


function getMethods(obj) {
    var result = [];
    for (var id in obj) {
      try {
        if (typeof(obj[id]) == "function") {
          result.push(id + ": " + obj[id].toString());
        }
      } catch (err) {
        result.push(id + ": inaccessible");
      }
    }
    return result;
}
