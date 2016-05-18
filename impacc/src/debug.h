#ifndef __IMPACC_DEBUG_H__
#define __IMPACC_DEBUG_H__

#include <stdio.h>
#include "runtime.h"

#define _CHECK_ENABLE
#define _DEBUG_ENABLE
//#define _TRACE_ENABLE
#define _INFO_ENABLE
#define _ERROR_ENABLE

#define COLOR_DEBUG_MSG

#ifdef COLOR_DEBUG_MSG
#define RED     "\033[22;31m"
#define GREEN   "\033[22;32m"
#define YELLOW  "\033[22;33m"
#define BLUE    "\033[22;34m"
#define PURPLE  "\033[22;35m"
#define CYAN    "\033[22;36m"
#define GRAY    "\033[22;37m"
#define RESET   "\e[m"
#else
#define RED     ""
#define GREEN   ""
#define YELLOW  ""
#define BLUE    ""
#define PURPLE  ""
#define CYAN    ""
#define GRAY    ""
#define RESET   ""
#endif

#ifdef _CHECK_ENABLE
#define _check() { printf(PURPLE "[C] [%d:%d:%s] [%s:%d:%s]" RESET "\n", gDE->ME(), gDE->rank, gDE->name, __FILE__, __LINE__, __func__); fflush(stdout); }
#else
#define _check()
#endif

#ifdef _DEBUG_ENABLE
#define _debug(fmt, ...) { printf(CYAN "[D] [%d:%d:%s] [%s:%d:%s] " fmt RESET "\n", gDE->ME(), gDE->rank, gDE->name, __FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define _debug(fmt, ...)
#endif

#ifdef _TRACE_ENABLE
#define _trace(fmt, ...) { printf(BLUE "[T] [%d:%d:%s] [%s:%d:%s] " fmt RESET "\n", gDE->ME(), gDE->rank, gDE->name, __FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define _trace(fmt, ...)
#endif

#ifdef _INFO_ENABLE
#define _info(fmt, ...) { printf(YELLOW "[I] [%d:%d:%s] [%s:%d:%s] " fmt RESET "\n", gDE->ME(), gDE->rank, gDE->name, __FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define _info(fmt, ...)
#endif

#ifdef _ERROR_ENABLE
#define _error(fmt, ...) { printf(RED "[E] [%d:%d:%s] [%s:%d:%s] " fmt RESET "\n", gDE->ME(), gDE->rank, gDE->name, __FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define _error(fmt, ...)
#endif

#endif /* __IMPACC_DEBUG_H__ */

