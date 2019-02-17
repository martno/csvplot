"use strict";


var LIMITS_BY_CATEGORY = {
    'pivot-category': {
        'rows': {'min': 0, 'max': Infinity},
        'columns': {'min': 0, 'max': Infinity},
        'values': {'min': 0, 'max': Infinity}
    },
    'catplot-category': {
        'rows': {'min': 0, 'max': 1},
        'columns': {'min': 1, 'max': 3},
        'values': {'min': 1, 'max': 1},
    }
};


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

    $('#apply').click(function() {
        var fieldData = {};
        for (let variable of ['columns', 'rows', 'values']) {
            fieldData[variable] = $('#field-' + variable).children('div').map(function(item) {
                return $(this).text();
            }).get();
        }

        var json_data = {
            'fieldData': fieldData,
            'formData': $('#field-form').serialize()
        };

        $('#apply').attr('disabled', true);

        var promise = postJson('/getresults', json_data);
        promise.done(function(content) {
            $('#main-window').html(content);
            $('#apply').attr('disabled', false);
        });
        promise.fail(function(content) {
            alert('[Error] ' + content.responseText);
            $('#apply').attr('disabled', false);
        });
    });


    $('input[type=radio][name=plot-category]').change(function() {
        updateFields();
    });

    updateFields();

    enableTooltips();
});


function updateFields() {
    $('.collection').find('.field').removeClass('transparent');

    for (let container of ['rows', 'columns', 'values']) {
        var fields = $(`#field-${container}`).find('.field').toArray();

        var plotCategory = getSelectedRadioButton('#plot-category', 'plot-category');
        var rangeByContainer = LIMITS_BY_CATEGORY[plotCategory];
        var range = rangeByContainer[container]
        var numOpaque = range.max;

        updateTransparentFields(fields, numOpaque);

        var rangeString = rangeToString(range);
        $(`#${container}-display`).html(rangeString);
    }
}


function enableTooltips() {
    $('[data-toggle="tooltip"]').tooltip();
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

    if (min === max) {
        return `${min}`;
    }
    return `${min}&rarr;${max}`
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
