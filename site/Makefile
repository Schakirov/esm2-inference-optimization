# Makefile for the ESM2-L4 project website (static site)
#
# Quick start (foreground dev server):
#   make dev
#
# Always-on, auto-restarting service (systemd --user, Restart=always):
#   make install   # install + enable + start the service
#   make status    # check it
#   make logs      # follow logs
#   make restart / make stop / make uninstall

PORT        ?= 8088
HOST        ?= 0.0.0.0
SITE_DIR    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SERVICE     := esm2-l4-site
UNIT_DIR    := $(HOME)/.config/systemd/user
UNIT_FILE   := $(UNIT_DIR)/$(SERVICE).service
PYTHON      ?= python3

.PHONY: dev install uninstall start stop restart status logs enable disable url

## Run a foreground dev server (Ctrl-C to stop)
dev:
	@echo "Serving $(SITE_DIR) at http://$(HOST):$(PORT)  (Ctrl-C to stop)"
	@cd "$(SITE_DIR)" && exec $(PYTHON) -m http.server $(PORT) --bind $(HOST)

## Install + enable + start the auto-restarting systemd --user service
install: $(UNIT_FILE)
	systemctl --user daemon-reload
	systemctl --user enable --now $(SERVICE).service
	@echo "Service started. http://$(HOST):$(PORT)"
	@systemctl --user --no-pager status $(SERVICE).service | head -n 6 || true

$(UNIT_FILE):
	@mkdir -p "$(UNIT_DIR)"
	@printf '%s\n' \
	  '[Unit]' \
	  'Description=ESM2-L4 project static website' \
	  'After=network.target' \
	  '' \
	  '[Service]' \
	  'Type=simple' \
	  'WorkingDirectory=$(SITE_DIR)' \
	  'ExecStart=$(PYTHON) -m http.server $(PORT) --bind $(HOST)' \
	  'Restart=always' \
	  'RestartSec=2' \
	  '' \
	  '[Install]' \
	  'WantedBy=default.target' \
	  > "$(UNIT_FILE)"
	@echo "Wrote $(UNIT_FILE)"

## Stop, disable, and remove the service
uninstall:
	-systemctl --user disable --now $(SERVICE).service
	-rm -f "$(UNIT_FILE)"
	systemctl --user daemon-reload
	@echo "Removed $(SERVICE).service"

start:   ; systemctl --user start $(SERVICE).service
stop:    ; systemctl --user stop $(SERVICE).service
restart: ; systemctl --user restart $(SERVICE).service
enable:  ; systemctl --user enable $(SERVICE).service
disable: ; systemctl --user disable $(SERVICE).service
status:  ; systemctl --user --no-pager status $(SERVICE).service
logs:    ; journalctl --user -u $(SERVICE).service -f
url:     ; @echo "http://$(HOST):$(PORT)"
