<script>

function showLoading(){
document.getElementById("loading").style = "visibility: visible";
}
function hideLoading(){
document.getElementById("loading").style = "visibility: hidden";
}

$( "form" ).submit(function( event ) {     
    //call show loading function here
    showLoading();
    $.ajax({
        type: "POST",
        url: "upload.php",
        enctype: 'multipart/form-data',
        data: {
            file: myfile
        },
        success: function () {
            //call hide function here
            hideLoading();
            alert("Data has been Uploaded: ");
        },
        error  : function (a) {//if an error occurs
            hideLoading();
            alert("An error occured while uploading data.\n error code : "+a.statusText);
        }
    });
});

</script>