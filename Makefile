# kllm — Makefile
#
# Targets:
#   make           Build the kllm CLI binary
#   make clean     Remove build artifacts
#   make install   Copy kllm binary to /usr/local/bin

UNAME := $(shell uname -s)

CC      := cc
CFLAGS  := -O3 -march=native -Wall -Wextra -Wno-unused-parameter

ifeq ($(UNAME),Darwin)
  LDFLAGS      := -lm -framework Accelerate
  CFLAGS       += -DACCELERATE_NEW_LAPACK
else
  LDFLAGS      := -lm -lopenblas
endif

CSRC     := csrc
PREFIX   ?= /usr/local

# ---- Targets ----

.PHONY: all clean install

all: kllm

# ---- Main CLI binary (kllm.c + kllm_compile.c + _tape_runner.c) ----
kllm: $(CSRC)/kllm.c $(CSRC)/kllm_compile.c $(CSRC)/_tape_runner.c $(CSRC)/tape_runner.h
	$(CC) $(CFLAGS) -o $@ $(CSRC)/kllm.c $(CSRC)/kllm_compile.c $(CSRC)/_tape_runner.c $(LDFLAGS)

# ---- Install ----
install: kllm
	install -d $(PREFIX)/bin
	install -m 755 kllm $(PREFIX)/bin/kllm

# ---- Clean ----
clean:
	rm -f kllm
	rm -rf $(CSRC)/*.dSYM
