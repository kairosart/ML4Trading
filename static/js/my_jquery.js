

$(document).ready(function(e){
    $("#submitButton").on('click',function(){

        $(this).val("Wait a second...");
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });
	$("#submitButton1").on('click',function(){
		$(this).html('Calculating...');
        $(this).css('color','white');
        $(this).css('background-color', 'red');
    });    
});