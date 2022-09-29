# Set the local var (if not set) from the environment var
function(set_local_from_env LOCAL_NAME ENV_NAME)
  if((NOT EXISTS ${LOCAL_NAME}) AND (DEFINED ENV{${ENV_NAME}}))
    set(${LOCAL_NAME} $ENV{${ENV_NAME}} PARENT_SCOPE)
  endif()
endfunction()
