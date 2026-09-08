	.file	"tsvc_2_vag.f90"
	.text
	.p2align 4
	.type	tsvc_2_vag_fp64._omp_fn.0, @function
tsvc_2_vag_fp64._omp_fn.0:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movq	%rdi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	call	omp_get_num_threads@PLT
	movl	%eax, %ebp
	call	omp_get_thread_num@PLT
	movslq	%eax, %rcx
	movq	(%rbx), %rax
	movslq	%ebp, %rsi
	cqto
	idivq	%rsi
	cmpq	%rdx, %rcx
	leaq	1(%rax), %rsi
	cmovl	%rsi, %rax
	movl	$0, %esi
	cmovl	%rsi, %rdx
	imulq	%rax, %rcx
	addq	%rcx, %rdx
	addq	%rdx, %rax
	cmpq	%rax, %rdx
	jge	.L7
	movq	8(%rbx), %r8
	movq	16(%rbx), %rdi
	movq	24(%rbx), %rsi
	.p2align 5
	.p2align 4
	.p2align 3
.L4:
	movslq	(%r8,%rdx,4), %rcx
	vmovsd	(%rdi,%rcx,8), %xmm0
	vmovsd	%xmm0, (%rsi,%rdx,8)
	incq	%rdx
	cmpq	%rdx, %rax
	jne	.L4
.L7:
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE1:
	.size	tsvc_2_vag_fp64._omp_fn.0, .-tsvc_2_vag_fp64._omp_fn.0
	.p2align 4
	.globl	tsvc_2_vag_fp64
	.type	tsvc_2_vag_fp64, @function
tsvc_2_vag_fp64:
.LFB0:
	.cfi_startproc
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	%rcx, (%rsp)
	movq	%rdx, 8(%rsp)
	movq	%rsi, 16(%rsp)
	xorl	%ecx, %ecx
	movq	%rdi, 24(%rsp)
	movq	%rsp, %rsi
	xorl	%edx, %edx
	leaq	tsvc_2_vag_fp64._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE0:
	.size	tsvc_2_vag_fp64, .-tsvc_2_vag_fp64
	.ident	"GCC: (Ubuntu 16-20260315-1ubuntu1~24~ppa1) 16.0.1 20260315 (experimental) [trunk r16-8100-g3aca3bae8ee]"
	.section	.note.GNU-stack,"",@progbits
