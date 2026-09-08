	.file	"ext_break_capture.f90"
	.text
	.p2align 4
	.globl	ext_break_capture_fp64
	.type	ext_break_capture_fp64, @function
ext_break_capture_fp64:
.LFB0:
	.cfi_startproc
	movq	.LC0(%rip), %rax
	movq	$-1, (%rsi)
	movq	%rax, (%rdx)
	testq	%rcx, %rcx
	jle	.L9
	vmovsd	.LC1(%rip), %xmm1
	xorl	%eax, %eax
	jmp	.L5
	.p2align 5
	.p2align 4
	.p2align 3
.L8:
	incq	%rax
	cmpq	%rax, %rcx
	je	.L9
.L5:
	vmovsd	(%rdi,%rax,8), %xmm0
	vcomisd	%xmm1, %xmm0
	jbe	.L8
	movq	%rax, (%rsi)
	vmovsd	%xmm0, (%rdx)
	ret
	.p2align 4
	.p2align 3
.L9:
	ret
	.cfi_endproc
.LFE0:
	.size	ext_break_capture_fp64, .-ext_break_capture_fp64
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	-1074790400
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.ident	"GCC: (Ubuntu 16-20260315-1ubuntu1~24~ppa1) 16.0.1 20260315 (experimental) [trunk r16-8100-g3aca3bae8ee]"
	.section	.note.GNU-stack,"",@progbits
