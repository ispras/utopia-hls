PATH_SETTINGS:=`pwd`/Settings.mak

all:
	@echo "\033[1;33m>>> Building DRIVER\033[0m"
	make -C driver PATH_SETTINGS=$(PATH_SETTINGS)
	@echo "\033[1;33m>>> Building API library\033[0m"
	make -C api PATH_SETTINGS=$(PATH_SETTINGS)
	@echo "\033[1;33m>>> Building TEST\033[0m"
	make -C test

clean:
	@echo "\033[1;33m>>> Cleaning DRIVER directory\033[0m"
	make -C driver clean PATH_SETTINGS=$(PATH_SETTINGS)
	@echo "\033[1;33m>>> Cleaning API directory\033[0m"
	make -C api clean PATH_SETTINGS=$(PATH_SETTINGS)
	@echo "\033[1;33m>>> Cleaning TEST directory\033[0m"
	make -C test clean 