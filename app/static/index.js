function upload(){
    // フォームデータを取得
    var formdata = new FormData($('#upload_form').get(0));

    // POSTでアップロード
    $.ajax({
        url  : "http://127.0.0.1:5000/upload",
        type : "POST",
        data : formdata,
        timeout: 5000,
        enctype: 'multipart/form-data',
        contentType : false,
        processData : false
    }).then(
        data => {
            console.log('success!'+ data);
            if (data.status === "ok") {
                $('#result').text("");
                $('#male').text(data.sex.male);
                $('#female').text(data.sex.female);
            } else {
                $('#result').text(data.status);
            }
        },
        error => console.log('Error : ' + error)
    );
}