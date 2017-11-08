

var resultList = [{"Break Owner " : "1", "Precision " : 1.00000, "Recall " : 1.00000, "Support " : 1239},
                  {"Break Owner " : "2", "Precision " : 0.98340, "Recall " : 1.00000, "Support " : 231},
                  {"Break Owner " : "3", "Precision " : 1.00000, "Recall " : 1.00000, "Support " : 487},
                  {"Break Owner " : "4", "Precision " : 0.00000, "Recall " : 1.00000, "Support " : 231},
                  {"Break Owner " : "5", "Precision " : 0.85714, "Recall " : 1.00000, "Support " : 231},
                  {"Break Owner " : "6", "Precision " : 1.00000, "Recall " : 1.00000, "Support " : 231},
                  {"Break Owner " : "7", "Precision " : 1.00000, "Recall " : 1.00000, "Support " : 231},
                  {"Break Owner " : "8", "Precision " : 1.00000, "Recall " : 1.00000, "Support " : 231}];

// Builds the HTML Table out of List.
function buildHtmlTable(selector) {
  var columns = addAllColumnHeaders(resultList, selector);

  for (var i = 0; i < resultList.length; i++) {
    var row$ = $('<tr/>');
    for (var colIndex = 0; colIndex < columns.length; colIndex++) {
      var cellValue = resultList[i][columns[colIndex]];
      if (cellValue == null) cellValue = "";
      row$.append($('<td/>').html(cellValue));
    }
    $(selector).append(row$);
  }
}

// Adds a header row to the table and returns the set of columns.
// Need to do union of keys from all records as some records may not contain
// all records.
function addAllColumnHeaders(resultList, selector) {
  var columnSet = [];
  var headerTr$ = $('<tr/>');

  for (var i = 0; i < resultList.length; i++) {
    var rowHash = resultList[i];
    for (var key in rowHash) {
      if ($.inArray(key, columnSet) == -1) {
        columnSet.push(key);
        headerTr$.append($('<th/>').html(key));
      }
    }
  }
  $(selector).append(headerTr$);

  return columnSet;
}