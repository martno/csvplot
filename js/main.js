"use strict";


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

    var promise = postJson('/getfields');
    promise.done(function(typeByField) {
        $.each(typeByField, function(field, type) {
            $('#field-fields').append(`<div class="${type} field">${field}</div>`);
        });
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
});


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
