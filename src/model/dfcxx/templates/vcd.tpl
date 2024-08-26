$date {{GEN_TIME}} $end
$version Utopia HLS $end
$timescale 1ps $end

$scope module logic $end
{{#VARS}}$var wire {{WIDTH}} {{NAME}} {{NAME}} $end
{{/VARS}}$upscope $end
$enddefinitions $end

$dumpvars
{{#INIT_VARS}}b{{INIT_VALUE}} {{NAME}}
{{/INIT_VARS}}$end

{{#TICKS}}#{{TICK}}
{{#VALUES}}b{{VALUE}} {{NAME}}
{{/VALUES}}
{{/TICKS}}#{{FINAL_TICK}}
