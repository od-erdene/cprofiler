function upload(){
    // フォームデータを取得
    var formdata = new FormData($('#upload_form').get(0));

    // POSTでアップロード
    $.ajax({
        // url  : "http://127.0.0.1:5000/api/upload",
        url  : "http://http://128.199.177.160/api/upload",
        type : "POST",
        data : formdata,
        timeout: 5000,
        enctype: 'multipart/form-data',
        contentType : false,
        processData : false
    }).then(
        data => {
            console.log(data);
            if (data.status === "ok") {
                $('#result').text("");
                $('#male').text(data.val.sex.male);
                $('#female').text(data.val.sex.female);
                $('#pincome').text(JSON.stringify(data.val.pincome));
                $('#hincome').text(JSON.stringify(data.val.hincome));
                $('#area').text(JSON.stringify(data.val.area));
                $('#married').text(JSON.stringify(data.val.married));
                $('#child').text(JSON.stringify(data.val.child));
                $('#prefecture').text(JSON.stringify(data.val.prefecture));
                $('#job').text(JSON.stringify(data.val.job));
                $('#student').text(JSON.stringify(data.val.student));
                $('#age').text(JSON.stringify(data.val.age));
            } else {
                $('#result').text(data.status);
                $('#male').text("");
                $('#female').text("");
                $('#pincome').text("");
                $('#hincome').text("");
                $('#area').text("");
                $('#married').text("");
                $('#child').text("");
                $('#prefecture').text("");
                $('#job').text("");
                $('#student').text("");
                $('#age').text("");
            }
        },
        error => console.log('Error : ' + error)
    );
}