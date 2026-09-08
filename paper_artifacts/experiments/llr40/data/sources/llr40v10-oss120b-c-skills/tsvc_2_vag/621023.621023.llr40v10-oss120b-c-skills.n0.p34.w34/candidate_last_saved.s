	.file	"tsvc_2_vag.c"
	.text
	.p2align 4
	.globl	tsvc_2_vag_fp64
	.type	tsvc_2_vag_fp64, @function
tsvc_2_vag_fp64:
.LFB0:
	.cfi_startproc
	endbr64
	testq	%rcx, %rcx
	jle	.L18
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	-1(%rcx), %rax
	movq	%rdi, %r9
	movq	%rdx, %r8
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	cmpq	$14, %rax
	jbe	.L6
	movq	%rcx, %rax
	movq	%rcx, -16(%rsp)
	xorl	%edx, %edx
	shrq	$4, %rax
	movq	%rax, %rdi
	movq	%rax, -8(%rsp)
	movq	%r8, %rax
	salq	$6, %rdi
	.p2align 4
	.p2align 3
.L4:
	leaq	(%rax,%rdx), %r10
	movslq	48(%r10), %r15
	movslq	52(%r10), %r14
	movslq	60(%r10), %r8
	movslq	56(%r10), %rcx
	movslq	32(%r10), %rbx
	movslq	36(%r10), %r11
	movslq	40(%r10), %r13
	movslq	44(%r10), %r12
	vmovsd	(%rsi,%r15,8), %xmm1
	vmovhpd	(%rsi,%r14,8), %xmm1, %xmm1
	movslq	16(%r10), %r15
	vmovsd	(%rsi,%rcx,8), %xmm0
	vmovhpd	(%rsi,%r8,8), %xmm0, %xmm0
	movslq	20(%r10), %r14
	vmovsd	(%rsi,%r13,8), %xmm2
	vmovhpd	(%rsi,%r12,8), %xmm2, %xmm2
	movslq	8(%r10), %r13
	movslq	12(%r10), %r12
	movslq	24(%r10), %r8
	vinsertf64x2	$0x1, %xmm0, %ymm1, %ymm1
	vmovsd	(%rsi,%rbx,8), %xmm0
	vmovhpd	(%rsi,%r11,8), %xmm0, %xmm0
	movslq	(%r10), %rbx
	movslq	4(%r10), %r11
	movslq	28(%r10), %r10
	vmovsd	(%rsi,%r13,8), %xmm3
	vmovhpd	(%rsi,%r12,8), %xmm3, %xmm3
	vinsertf64x2	$0x1, %xmm2, %ymm0, %ymm0
	vmovsd	(%rsi,%r15,8), %xmm2
	vmovhpd	(%rsi,%r14,8), %xmm2, %xmm2
	vinsertf64x4	$0x1, %ymm1, %zmm0, %zmm0
	vmovsd	(%rsi,%r8,8), %xmm1
	vmovhpd	(%rsi,%r10,8), %xmm1, %xmm1
	vinsertf64x2	$0x1, %xmm1, %ymm2, %ymm2
	vmovsd	(%rsi,%rbx,8), %xmm1
	vmovhpd	(%rsi,%r11,8), %xmm1, %xmm1
	vinsertf64x2	$0x1, %xmm3, %ymm1, %ymm1
	vinsertf64x4	$0x1, %ymm2, %zmm1, %zmm1
	vmovupd	%zmm1, (%r9,%rdx,2)
	vmovupd	%zmm0, 64(%r9,%rdx,2)
	addq	$64, %rdx
	cmpq	%rdx, %rdi
	jne	.L4
	movq	%rax, %r8
	movq	-8(%rsp), %rax
	movq	-16(%rsp), %rcx
	salq	$4, %rax
	movq	%rax, %r10
	cmpq	%rax, %rcx
	je	.L15
.L3:
	movq	%rcx, %rdi
	subq	%r10, %rdi
	leaq	-1(%rdi), %rdx
	cmpq	$6, %rdx
	jbe	.L5
	leaq	(%r8,%r10,4), %rdx
	leaq	(%r9,%r10,8), %r13
	movslq	16(%rdx), %r11
	movslq	20(%rdx), %r10
	movslq	28(%rdx), %rbx
	movslq	24(%rdx), %r12
	vmovsd	(%rsi,%r11,8), %xmm0
	vmovhpd	(%rsi,%r10,8), %xmm0, %xmm0
	movslq	(%rdx), %r11
	vmovsd	(%rsi,%r12,8), %xmm1
	vmovhpd	(%rsi,%rbx,8), %xmm1, %xmm1
	movslq	4(%rdx), %r10
	movslq	8(%rdx), %rbx
	movslq	12(%rdx), %rdx
	vmovsd	(%rsi,%rbx,8), %xmm2
	vinsertf64x2	$0x1, %xmm1, %ymm0, %ymm0
	vmovhpd	(%rsi,%rdx,8), %xmm2, %xmm2
	vmovsd	(%rsi,%r11,8), %xmm1
	vmovhpd	(%rsi,%r10,8), %xmm1, %xmm1
	movq	%rdi, %rdx
	vmovupd	%ymm0, 32(%r13)
	andq	$-8, %rdx
	andl	$7, %edi
	vinsertf64x2	$0x1, %xmm2, %ymm1, %ymm1
	vmovupd	%ymm1, 0(%r13)
	je	.L15
	addq	%rdx, %rax
.L5:
	movslq	(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	1(%rax), %rdi
	vmovsd	%xmm0, (%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	4(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	2(%rax), %rdi
	vmovsd	%xmm0, 8(%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	8(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	3(%rax), %rdi
	vmovsd	%xmm0, 16(%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	12(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	4(%rax), %rdi
	vmovsd	%xmm0, 24(%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	16(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	5(%rax), %rdi
	vmovsd	%xmm0, 32(%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	20(%r8,%rax,4), %rdi
	vmovsd	(%rsi,%rdi,8), %xmm0
	leaq	6(%rax), %rdi
	vmovsd	%xmm0, 40(%r9,%rax,8)
	cmpq	%rdi, %rcx
	jle	.L15
	movslq	24(%r8,%rax,4), %rdx
	vmovsd	(%rsi,%rdx,8), %xmm0
	vmovsd	%xmm0, 48(%r9,%rax,8)
	vzeroupper
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4
	.p2align 3
.L15:
	.cfi_restore_state
	vzeroupper
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4
	.p2align 3
.L18:
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	.cfi_restore 15
	ret
	.p2align 4
	.p2align 3
.L6:
	.cfi_def_cfa 6, 16
	.cfi_offset 3, -56
	.cfi_offset 6, -16
	.cfi_offset 12, -48
	.cfi_offset 13, -40
	.cfi_offset 14, -32
	.cfi_offset 15, -24
	xorl	%r10d, %r10d
	xorl	%eax, %eax
	jmp	.L3
	.cfi_endproc
.LFE0:
	.size	tsvc_2_vag_fp64, .-tsvc_2_vag_fp64
	.ident	"GCC: (Ubuntu 16-20260315-1ubuntu1~24~ppa1) 16.0.1 20260315 (experimental) [trunk r16-8100-g3aca3bae8ee]"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
