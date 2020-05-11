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
            console.log('test!'+ data.status);
            console.log('test!'+ data["status"]);
            if (data.status === "ok") {
                $('#result').text("");
                $('#male').text(data.val.sex.male);
                $('#female').text(data.val.sex.female);
            } else {
                $('#result').text(data.status);
            }
        },
        error => console.log('Error : ' + error)
    );
}