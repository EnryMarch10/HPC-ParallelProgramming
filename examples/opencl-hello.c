/****************************************************************************
 *
 * opencl-hello.c - Hello world with OpenCL
 *
 * Copyright (C) 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * cc opencl-hello.c simpleCL.c -o opencl-hello -lOpenCL
 *
 * Run with:
 * ./opencl-hello
 *
 ****************************************************************************/
#include <stdio.h>
#include "simpleCL.h"

int main(void)
{
    sclInitFromString("__kernel void mykernel(void) { }");
    sclSetArgsLaunchKernel(sclCreateKernel("mykernel"),
                           DIM1(1), DIM1(1),
                           NULL);
    printf("Hello World!\n");
    sclFinalize();
    return 0;
}
