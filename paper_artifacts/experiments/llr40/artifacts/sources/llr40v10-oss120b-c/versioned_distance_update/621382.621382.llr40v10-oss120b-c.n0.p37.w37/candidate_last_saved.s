	.file	"versioned_distance_update.c"
	.text
	.p2align 4
	.type	versioned_distance_update_fp64._omp_fn.0, @function
versioned_distance_update_fp64._omp_fn.0:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbp
	movq	32(%rdi), %rbx
	call	omp_get_num_threads@PLT
	movl	%eax, %r12d
	call	omp_get_thread_num@PLT
	movslq	%eax, %r8
	movslq	%r12d, %rcx
	movq	%rbx, %rax
	cqto
	idivq	%rcx
	cmpq	%rdx, %r8
	leaq	1(%rax), %r9
	cmovge	%rax, %r9
	movl	$0, %eax
	cmovl	%rax, %rdx
	imulq	%r9, %r8
	addq	%rdx, %r8
	addq	%r8, %r9
	cmpq	%r9, %r8
	jge	.L19
	vmovsd	.LC0(%rip), %xmm2
	movq	24(%rbp), %rdx
	movq	16(%rbp), %rsi
	movq	8(%rbp), %rdi
	movq	0(%rbp), %rcx
	cmpq	$1, %rbx
	jne	.L22
	.p2align 4
	.p2align 3
.L4:
	incq	%r8
	cmpq	%r8, %rdx
	jle	.L7
	vmovsd	-8(%rcx,%r8,8), %xmm1
	movq	%r8, %rax
	.p2align 6
	.p2align 4
	.p2align 3
.L8:
	vmovsd	(%rsi,%rax,8), %xmm0
	vmulsd	(%rdi,%rax,8), %xmm0, %xmm0
	vfmadd132sd	%xmm2, %xmm0, %xmm1
	vmovsd	%xmm1, (%rcx,%rax,8)
	incq	%rax
	cmpq	%rax, %rdx
	jne	.L8
.L7:
	cmpq	%r8, %r9
	jne	.L4
.L19:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L22:
	.cfi_restore_state
	movq	%rbx, %rax
	addq	%rbx, %r8
	addq	%rbx, %r9
	vmovapd	%xmm2, %xmm1
	negq	%rax
	leaq	(%rcx,%rax,8), %r10
	.p2align 4
	.p2align 3
.L6:
	movq	%r8, %rax
	cmpq	%r8, %rdx
	jle	.L9
	.p2align 6
	.p2align 4
	.p2align 3
.L5:
	vmovsd	(%rdi,%rax,8), %xmm0
	vmulsd	(%rsi,%rax,8), %xmm0, %xmm0
	vfmadd231sd	(%r10,%rax,8), %xmm1, %xmm0
	vmovsd	%xmm0, (%rcx,%rax,8)
	addq	%rbx, %rax
	cmpq	%rax, %rdx
	jg	.L5
.L9:
	incq	%r8
	cmpq	%r8, %r9
	jne	.L6
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE1:
	.size	versioned_distance_update_fp64._omp_fn.0, .-versioned_distance_update_fp64._omp_fn.0
	.p2align 4
	.globl	versioned_distance_update_fp64
	.type	versioned_distance_update_fp64, @function
versioned_distance_update_fp64:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsi, %r11
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%rbx
	andq	$-64, %rsp
	subq	$64, %rsp
	.cfi_offset 14, -24
	.cfi_offset 3, -32
	movq	%fs:40, %rbx
	movq	%rbx, 56(%rsp)
	movq	%rdx, %rbx
	cmpq	$1, %r8
	jg	.L24
	cmpq	%rcx, %r8
	jge	.L23
	leaq	-8(,%r8,8), %rax
	cmpq	$48, %rax
	jbe	.L26
	subq	%r8, %rcx
	vbroadcastsd	.LC0(%rip), %zmm1
	leaq	-1(%rcx), %rax
	cmpq	$6, %rax
	jbe	.L31
	movq	%rcx, %rax
	leaq	0(,%r8,8), %rdx
	shrq	$3, %rax
	movq	%rax, %r14
	leaq	(%rsi,%rdx), %r10
	leaq	(%rbx,%rdx), %r9
	addq	%rdi, %rdx
	salq	$6, %rax
	movq	%rax, %rsi
	xorl	%eax, %eax
	.p2align 6
	.p2align 4
	.p2align 3
.L28:
	vmovupd	(%r9,%rax), %zmm0
	vmulpd	(%r10,%rax), %zmm0, %zmm0
	vfmadd231pd	(%rdi,%rax), %zmm1, %zmm0
	vmovupd	%zmm0, (%rdx,%rax)
	addq	$64, %rax
	cmpq	%rsi, %rax
	jne	.L28
	leaq	0(,%r14,8), %rax
	cmpq	%rax, %rcx
	je	.L36
.L27:
	subq	%rax, %rcx
	addq	%rax, %r8
	vpbroadcastq	%rcx, %zmm0
	vpcmpuq	$6, .LC2(%rip), %zmm0, %k1
	vmovupd	(%r11,%r8,8), %zmm3{%k1}{z}
	vmovupd	(%rbx,%r8,8), %zmm2{%k1}{z}
	vmovupd	(%rdi,%rax,8), %zmm0{%k1}{z}
	vmulpd	%zmm3, %zmm2, %zmm2
	vfmadd132pd	%zmm1, %zmm2, %zmm0
	vmovupd	%zmm0, (%rdi,%r8,8){%k1}
.L36:
	vzeroupper
.L23:
	movq	56(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L37
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4
	.p2align 3
.L24:
	.cfi_restore_state
	movq	%rcx, 40(%rsp)
	movq	%rsi, 24(%rsp)
	movq	%rdi, 16(%rsp)
	leaq	16(%rsp), %rsi
	xorl	%ecx, %ecx
	movq	%r8, 48(%rsp)
	movq	%rbx, 32(%rsp)
	xorl	%edx, %edx
	leaq	versioned_distance_update_fp64._omp_fn.0(%rip), %rdi
	call	GOMP_parallel@PLT
	jmp	.L23
	.p2align 4
	.p2align 3
.L26:
	vmovsd	.LC0(%rip), %xmm1
	movq	%r8, %rax
	negq	%rax
	leaq	(%rdi,%rax,8), %rax
	.p2align 6
	.p2align 4
	.p2align 3
.L29:
	vmovsd	(%rbx,%r8,8), %xmm0
	vmulsd	(%r11,%r8,8), %xmm0, %xmm0
	vfmadd231sd	(%rax,%r8,8), %xmm1, %xmm0
	vmovsd	%xmm0, (%rdi,%r8,8)
	incq	%r8
	cmpq	%r8, %rcx
	jne	.L29
	jmp	.L23
.L31:
	xorl	%eax, %eax
	jmp	.L27
.L37:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE0:
	.size	versioned_distance_update_fp64, .-versioned_distance_update_fp64
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1072168960
	.section	.rodata
	.align 64
.LC2:
	.quad	0
	.quad	1
	.quad	2
	.quad	3
	.quad	4
	.quad	5
	.quad	6
	.quad	7
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
