
a.out:     file format elf64-x86-64


Disassembly of section .init:

00000000000008a0 <_init>:
 8a0:	48 83 ec 08          	sub    $0x8,%rsp
 8a4:	48 8b 05 2d 17 20 00 	mov    0x20172d(%rip),%rax        # 201fd8 <__gmon_start__>
 8ab:	48 85 c0             	test   %rax,%rax
 8ae:	74 02                	je     8b2 <_init+0x12>
 8b0:	ff d0                	callq  *%rax
 8b2:	48 83 c4 08          	add    $0x8,%rsp
 8b6:	c3                   	retq   

Disassembly of section .plt:

00000000000008c0 <.plt>:
 8c0:	ff 35 b2 16 20 00    	pushq  0x2016b2(%rip)        # 201f78 <_GLOBAL_OFFSET_TABLE_+0x8>
 8c6:	ff 25 b4 16 20 00    	jmpq   *0x2016b4(%rip)        # 201f80 <_GLOBAL_OFFSET_TABLE_+0x10>
 8cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000008d0 <puts@plt>:
 8d0:	ff 25 b2 16 20 00    	jmpq   *0x2016b2(%rip)        # 201f88 <puts@GLIBC_2.2.5>
 8d6:	68 00 00 00 00       	pushq  $0x0
 8db:	e9 e0 ff ff ff       	jmpq   8c0 <.plt>

00000000000008e0 <exit@plt>:
 8e0:	ff 25 aa 16 20 00    	jmpq   *0x2016aa(%rip)        # 201f90 <exit@GLIBC_2.2.5>
 8e6:	68 01 00 00 00       	pushq  $0x1
 8eb:	e9 d0 ff ff ff       	jmpq   8c0 <.plt>

00000000000008f0 <__printf_chk@plt>:
 8f0:	ff 25 a2 16 20 00    	jmpq   *0x2016a2(%rip)        # 201f98 <__printf_chk@GLIBC_2.3.4>
 8f6:	68 02 00 00 00       	pushq  $0x2
 8fb:	e9 c0 ff ff ff       	jmpq   8c0 <.plt>

0000000000000900 <aligned_alloc@plt>:
 900:	ff 25 9a 16 20 00    	jmpq   *0x20169a(%rip)        # 201fa0 <aligned_alloc@GLIBC_2.16>
 906:	68 03 00 00 00       	pushq  $0x3
 90b:	e9 b0 ff ff ff       	jmpq   8c0 <.plt>

0000000000000910 <pthread_create@plt>:
 910:	ff 25 92 16 20 00    	jmpq   *0x201692(%rip)        # 201fa8 <pthread_create@GLIBC_2.2.5>
 916:	68 04 00 00 00       	pushq  $0x4
 91b:	e9 a0 ff ff ff       	jmpq   8c0 <.plt>

0000000000000920 <pthread_join@plt>:
 920:	ff 25 8a 16 20 00    	jmpq   *0x20168a(%rip)        # 201fb0 <pthread_join@GLIBC_2.2.5>
 926:	68 05 00 00 00       	pushq  $0x5
 92b:	e9 90 ff ff ff       	jmpq   8c0 <.plt>

0000000000000930 <cblas_cgemm@plt>:
 930:	ff 25 82 16 20 00    	jmpq   *0x201682(%rip)        # 201fb8 <cblas_cgemm>
 936:	68 06 00 00 00       	pushq  $0x6
 93b:	e9 80 ff ff ff       	jmpq   8c0 <.plt>

0000000000000940 <__stack_chk_fail@plt>:
 940:	ff 25 7a 16 20 00    	jmpq   *0x20167a(%rip)        # 201fc0 <__stack_chk_fail@GLIBC_2.4>
 946:	68 07 00 00 00       	pushq  $0x7
 94b:	e9 70 ff ff ff       	jmpq   8c0 <.plt>

0000000000000950 <rand@plt>:
 950:	ff 25 72 16 20 00    	jmpq   *0x201672(%rip)        # 201fc8 <rand@GLIBC_2.2.5>
 956:	68 08 00 00 00       	pushq  $0x8
 95b:	e9 60 ff ff ff       	jmpq   8c0 <.plt>

0000000000000960 <fflush@plt>:
 960:	ff 25 6a 16 20 00    	jmpq   *0x20166a(%rip)        # 201fd0 <fflush@GLIBC_2.2.5>
 966:	68 09 00 00 00       	pushq  $0x9
 96b:	e9 50 ff ff ff       	jmpq   8c0 <.plt>

Disassembly of section .plt.got:

0000000000000970 <__cxa_finalize@plt>:
 970:	ff 25 82 16 20 00    	jmpq   *0x201682(%rip)        # 201ff8 <__cxa_finalize@GLIBC_2.2.5>
 976:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000000980 <main>:
     980:	41 57                	push   %r15
     982:	41 56                	push   %r14
     984:	be 00 e4 25 00       	mov    $0x25e400,%esi
     989:	41 55                	push   %r13
     98b:	41 54                	push   %r12
     98d:	bf 40 00 00 00       	mov    $0x40,%edi
     992:	55                   	push   %rbp
     993:	53                   	push   %rbx
     994:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
     99b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
     9a2:	00 00 
     9a4:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
     9ab:	00 
     9ac:	31 c0                	xor    %eax,%eax
     9ae:	e8 4d ff ff ff       	callq  900 <aligned_alloc@plt>
     9b3:	48 8d a8 00 ca 03 00 	lea    0x3ca00(%rax),%rbp
     9ba:	49 89 c4             	mov    %rax,%r12
     9bd:	48 89 c3             	mov    %rax,%rbx
     9c0:	e8 8b ff ff ff       	callq  950 <rand@plt>
     9c5:	66 0f ef c0          	pxor   %xmm0,%xmm0
     9c9:	48 83 c3 04          	add    $0x4,%rbx
     9cd:	f3 0f 2a c0          	cvtsi2ss %eax,%xmm0
     9d1:	f3 0f 59 05 8f 09 00 	mulss  0x98f(%rip),%xmm0        # 1368 <_IO_stdin_used+0xf8>
     9d8:	00 
     9d9:	f3 0f 11 43 fc       	movss  %xmm0,-0x4(%rbx)
     9de:	48 39 dd             	cmp    %rbx,%rbp
     9e1:	75 dd                	jne    9c0 <main+0x40>
     9e3:	be 00 13 00 00       	mov    $0x1300,%esi
     9e8:	bf 40 00 00 00       	mov    $0x40,%edi
     9ed:	e8 0e ff ff ff       	callq  900 <aligned_alloc@plt>
     9f2:	be 00 13 00 00       	mov    $0x1300,%esi
     9f7:	bf 40 00 00 00       	mov    $0x40,%edi
     9fc:	48 89 c5             	mov    %rax,%rbp
     9ff:	e8 fc fe ff ff       	callq  900 <aligned_alloc@plt>
     a04:	be 00 e0 7f 04       	mov    $0x47fe000,%esi
     a09:	bf 40 00 00 00       	mov    $0x40,%edi
     a0e:	49 89 c6             	mov    %rax,%r14
     a11:	e8 ea fe ff ff       	callq  900 <aligned_alloc@plt>
     a16:	be 00 e0 7f 04       	mov    $0x47fe000,%esi
     a1b:	bf 40 00 00 00       	mov    $0x40,%edi
     a20:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
     a25:	e8 d6 fe ff ff       	callq  900 <aligned_alloc@plt>
     a2a:	be 00 94 07 00       	mov    $0x79400,%esi
     a2f:	bf 40 00 00 00       	mov    $0x40,%edi
     a34:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
     a39:	e8 c2 fe ff ff       	callq  900 <aligned_alloc@plt>
     a3e:	be 00 94 07 00       	mov    $0x79400,%esi
     a43:	bf 40 00 00 00       	mov    $0x40,%edi
     a48:	49 89 c7             	mov    %rax,%r15
     a4b:	e8 b0 fe ff ff       	callq  900 <aligned_alloc@plt>
     a50:	be 00 c8 4b 00       	mov    $0x4bc800,%esi
     a55:	bf 40 00 00 00       	mov    $0x40,%edi
     a5a:	49 89 c5             	mov    %rax,%r13
     a5d:	e8 9e fe ff ff       	callq  900 <aligned_alloc@plt>
     a62:	be 00 c8 4b 00       	mov    $0x4bc800,%esi
     a67:	bf 40 00 00 00       	mov    $0x40,%edi
     a6c:	48 89 c3             	mov    %rax,%rbx
     a6f:	e8 8c fe ff ff       	callq  900 <aligned_alloc@plt>
     a74:	31 d2                	xor    %edx,%edx
     a76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     a7d:	00 00 00 
     a80:	41 0f 28 04 14       	movaps (%r12,%rdx,1),%xmm0
     a85:	41 0f 28 94 14 00 e5 	movaps 0x1e500(%r12,%rdx,1),%xmm2
     a8c:	01 00 
     a8e:	0f 28 c8             	movaps %xmm0,%xmm1
     a91:	0f 15 c2             	unpckhps %xmm2,%xmm0
     a94:	0f 14 ca             	unpcklps %xmm2,%xmm1
     a97:	0f 29 44 53 10       	movaps %xmm0,0x10(%rbx,%rdx,2)
     a9c:	0f 29 0c 53          	movaps %xmm1,(%rbx,%rdx,2)
     aa0:	0f 29 0c 50          	movaps %xmm1,(%rax,%rdx,2)
     aa4:	0f 29 44 50 10       	movaps %xmm0,0x10(%rax,%rdx,2)
     aa9:	48 83 c2 10          	add    $0x10,%rdx
     aad:	48 81 fa 00 e5 01 00 	cmp    $0x1e500,%rdx
     ab4:	75 ca                	jne    a80 <main+0x100>
     ab6:	48 89 6c 24 18       	mov    %rbp,0x18(%rsp)
     abb:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
     ac0:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
     ac5:	f3 0f 7e 44 24 18    	movq   0x18(%rsp),%xmm0
     acb:	48 8d 15 2e 04 00 00 	lea    0x42e(%rip),%rdx        # f00 <calcThreadRowmajor>
     ad2:	0f 16 44 24 08       	movhps 0x8(%rsp),%xmm0
     ad7:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
     adc:	31 f6                	xor    %esi,%esi
     ade:	66 0f 6f 0d 8a 08 00 	movdqa 0x88a(%rip),%xmm1        # 1370 <_IO_stdin_used+0x100>
     ae5:	00 
     ae6:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
     aeb:	0f 29 4c 24 50       	movaps %xmm1,0x50(%rsp)
     af0:	0f 29 8c 24 80 00 00 	movaps %xmm1,0x80(%rsp)
     af7:	00 
     af8:	f3 0f 7e 44 24 08    	movq   0x8(%rsp),%xmm0
     afe:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
     b03:	0f 16 44 24 08       	movhps 0x8(%rsp),%xmm0
     b08:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
     b0d:	0f 29 44 24 40       	movaps %xmm0,0x40(%rsp)
     b12:	f3 0f 7e 44 24 08    	movq   0x8(%rsp),%xmm0
     b18:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
     b1d:	0f 16 44 24 10       	movhps 0x10(%rsp),%xmm0
     b22:	0f 29 44 24 60       	movaps %xmm0,0x60(%rsp)
     b27:	f3 0f 7e 44 24 08    	movq   0x8(%rsp),%xmm0
     b2d:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
     b32:	0f 16 44 24 08       	movhps 0x8(%rsp),%xmm0
     b37:	0f 29 44 24 70       	movaps %xmm0,0x70(%rsp)
     b3c:	e8 cf fd ff ff       	callq  910 <pthread_create@plt>
     b41:	85 c0                	test   %eax,%eax
     b43:	75 67                	jne    bac <main+0x22c>
     b45:	48 8d 4c 24 60       	lea    0x60(%rsp),%rcx
     b4a:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
     b4f:	48 8d 15 ca 01 00 00 	lea    0x1ca(%rip),%rdx        # d20 <calcThreadColmajor>
     b56:	31 f6                	xor    %esi,%esi
     b58:	e8 b3 fd ff ff       	callq  910 <pthread_create@plt>
     b5d:	85 c0                	test   %eax,%eax
     b5f:	0f 85 8e 00 00 00    	jne    bf3 <main+0x273>
     b65:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
     b6a:	31 f6                	xor    %esi,%esi
     b6c:	e8 af fd ff ff       	callq  920 <pthread_join@plt>
     b71:	85 c0                	test   %eax,%eax
     b73:	75 68                	jne    bdd <main+0x25d>
     b75:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
     b7a:	31 f6                	xor    %esi,%esi
     b7c:	e8 9f fd ff ff       	callq  920 <pthread_join@plt>
     b81:	85 c0                	test   %eax,%eax
     b83:	75 42                	jne    bc7 <main+0x247>
     b85:	31 c0                	xor    %eax,%eax
     b87:	48 8b 8c 24 98 00 00 	mov    0x98(%rsp),%rcx
     b8e:	00 
     b8f:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
     b96:	00 00 
     b98:	75 28                	jne    bc2 <main+0x242>
     b9a:	48 81 c4 a8 00 00 00 	add    $0xa8,%rsp
     ba1:	5b                   	pop    %rbx
     ba2:	5d                   	pop    %rbp
     ba3:	41 5c                	pop    %r12
     ba5:	41 5d                	pop    %r13
     ba7:	41 5e                	pop    %r14
     ba9:	41 5f                	pop    %r15
     bab:	c3                   	retq   
     bac:	48 8d 3d f5 06 00 00 	lea    0x6f5(%rip),%rdi        # 12a8 <_IO_stdin_used+0x38>
     bb3:	e8 18 fd ff ff       	callq  8d0 <puts@plt>
     bb8:	bf 01 00 00 00       	mov    $0x1,%edi
     bbd:	e8 1e fd ff ff       	callq  8e0 <exit@plt>
     bc2:	e8 79 fd ff ff       	callq  940 <__stack_chk_fail@plt>
     bc7:	48 8d 3d 6a 07 00 00 	lea    0x76a(%rip),%rdi        # 1338 <_IO_stdin_used+0xc8>
     bce:	e8 fd fc ff ff       	callq  8d0 <puts@plt>
     bd3:	bf 01 00 00 00       	mov    $0x1,%edi
     bd8:	e8 03 fd ff ff       	callq  8e0 <exit@plt>
     bdd:	48 8d 3d 24 07 00 00 	lea    0x724(%rip),%rdi        # 1308 <_IO_stdin_used+0x98>
     be4:	e8 e7 fc ff ff       	callq  8d0 <puts@plt>
     be9:	bf 01 00 00 00       	mov    $0x1,%edi
     bee:	e8 ed fc ff ff       	callq  8e0 <exit@plt>
     bf3:	48 8d 3d de 06 00 00 	lea    0x6de(%rip),%rdi        # 12d8 <_IO_stdin_used+0x68>
     bfa:	e8 d1 fc ff ff       	callq  8d0 <puts@plt>
     bff:	bf 01 00 00 00       	mov    $0x1,%edi
     c04:	e8 d7 fc ff ff       	callq  8e0 <exit@plt>
     c09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000000c10 <_start>:
     c10:	31 ed                	xor    %ebp,%ebp
     c12:	49 89 d1             	mov    %rdx,%r9
     c15:	5e                   	pop    %rsi
     c16:	48 89 e2             	mov    %rsp,%rdx
     c19:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
     c1d:	50                   	push   %rax
     c1e:	54                   	push   %rsp
     c1f:	4c 8d 05 3a 06 00 00 	lea    0x63a(%rip),%r8        # 1260 <__libc_csu_fini>
     c26:	48 8d 0d c3 05 00 00 	lea    0x5c3(%rip),%rcx        # 11f0 <__libc_csu_init>
     c2d:	48 8d 3d 4c fd ff ff 	lea    -0x2b4(%rip),%rdi        # 980 <main>
     c34:	ff 15 a6 13 20 00    	callq  *0x2013a6(%rip)        # 201fe0 <__libc_start_main@GLIBC_2.2.5>
     c3a:	f4                   	hlt    
     c3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000000c40 <deregister_tm_clones>:
     c40:	48 8d 3d d1 13 20 00 	lea    0x2013d1(%rip),%rdi        # 202018 <stdout@@GLIBC_2.2.5>
     c47:	55                   	push   %rbp
     c48:	48 8d 05 c9 13 20 00 	lea    0x2013c9(%rip),%rax        # 202018 <stdout@@GLIBC_2.2.5>
     c4f:	48 39 f8             	cmp    %rdi,%rax
     c52:	48 89 e5             	mov    %rsp,%rbp
     c55:	74 19                	je     c70 <deregister_tm_clones+0x30>
     c57:	48 8b 05 8a 13 20 00 	mov    0x20138a(%rip),%rax        # 201fe8 <_ITM_deregisterTMCloneTable>
     c5e:	48 85 c0             	test   %rax,%rax
     c61:	74 0d                	je     c70 <deregister_tm_clones+0x30>
     c63:	5d                   	pop    %rbp
     c64:	ff e0                	jmpq   *%rax
     c66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     c6d:	00 00 00 
     c70:	5d                   	pop    %rbp
     c71:	c3                   	retq   
     c72:	0f 1f 40 00          	nopl   0x0(%rax)
     c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     c7d:	00 00 00 

0000000000000c80 <register_tm_clones>:
     c80:	48 8d 3d 91 13 20 00 	lea    0x201391(%rip),%rdi        # 202018 <stdout@@GLIBC_2.2.5>
     c87:	48 8d 35 8a 13 20 00 	lea    0x20138a(%rip),%rsi        # 202018 <stdout@@GLIBC_2.2.5>
     c8e:	55                   	push   %rbp
     c8f:	48 29 fe             	sub    %rdi,%rsi
     c92:	48 89 e5             	mov    %rsp,%rbp
     c95:	48 c1 fe 03          	sar    $0x3,%rsi
     c99:	48 89 f0             	mov    %rsi,%rax
     c9c:	48 c1 e8 3f          	shr    $0x3f,%rax
     ca0:	48 01 c6             	add    %rax,%rsi
     ca3:	48 d1 fe             	sar    %rsi
     ca6:	74 18                	je     cc0 <register_tm_clones+0x40>
     ca8:	48 8b 05 41 13 20 00 	mov    0x201341(%rip),%rax        # 201ff0 <_ITM_registerTMCloneTable>
     caf:	48 85 c0             	test   %rax,%rax
     cb2:	74 0c                	je     cc0 <register_tm_clones+0x40>
     cb4:	5d                   	pop    %rbp
     cb5:	ff e0                	jmpq   *%rax
     cb7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
     cbe:	00 00 
     cc0:	5d                   	pop    %rbp
     cc1:	c3                   	retq   
     cc2:	0f 1f 40 00          	nopl   0x0(%rax)
     cc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     ccd:	00 00 00 

0000000000000cd0 <__do_global_dtors_aux>:
     cd0:	80 3d 49 13 20 00 00 	cmpb   $0x0,0x201349(%rip)        # 202020 <completed.7698>
     cd7:	75 2f                	jne    d08 <__do_global_dtors_aux+0x38>
     cd9:	48 83 3d 17 13 20 00 	cmpq   $0x0,0x201317(%rip)        # 201ff8 <__cxa_finalize@GLIBC_2.2.5>
     ce0:	00 
     ce1:	55                   	push   %rbp
     ce2:	48 89 e5             	mov    %rsp,%rbp
     ce5:	74 0c                	je     cf3 <__do_global_dtors_aux+0x23>
     ce7:	48 8b 3d 1a 13 20 00 	mov    0x20131a(%rip),%rdi        # 202008 <__dso_handle>
     cee:	e8 7d fc ff ff       	callq  970 <__cxa_finalize@plt>
     cf3:	e8 48 ff ff ff       	callq  c40 <deregister_tm_clones>
     cf8:	c6 05 21 13 20 00 01 	movb   $0x1,0x201321(%rip)        # 202020 <completed.7698>
     cff:	5d                   	pop    %rbp
     d00:	c3                   	retq   
     d01:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
     d08:	f3 c3                	repz retq 
     d0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000000d10 <frame_dummy>:
     d10:	55                   	push   %rbp
     d11:	48 89 e5             	mov    %rsp,%rbp
     d14:	5d                   	pop    %rbp
     d15:	e9 66 ff ff ff       	jmpq   c80 <register_tm_clones>
     d1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000000d20 <calcThreadColmajor>:
     d20:	41 57                	push   %r15
     d22:	41 56                	push   %r14
     d24:	41 55                	push   %r13
     d26:	41 54                	push   %r12
     d28:	45 31 ed             	xor    %r13d,%r13d
     d2b:	55                   	push   %rbp
     d2c:	53                   	push   %rbx
     d2d:	48 89 fb             	mov    %rdi,%rbx
     d30:	48 83 ec 38          	sub    $0x38,%rsp
     d34:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
     d3b:	00 00 
     d3d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
     d42:	31 c0                	xor    %eax,%eax
     d44:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
     d49:	48 8d 6c 24 20       	lea    0x20(%rsp),%rbp
     d4e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
     d53:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     d58:	8b 4b 24             	mov    0x24(%rbx),%ecx
     d5b:	44 8b 4b 2c          	mov    0x2c(%rbx),%r9d
     d5f:	44 8b 43 28          	mov    0x28(%rbx),%r8d
     d63:	8b 35 bb 12 20 00    	mov    0x2012bb(%rip),%esi        # 202024 <seed.5154>
     d69:	4c 8b 33             	mov    (%rbx),%r14
     d6c:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
     d70:	89 c8                	mov    %ecx,%eax
     d72:	41 0f af c1          	imul   %r9d,%eax
     d76:	43 8d 3c 80          	lea    (%r8,%r8,4),%edi
     d7a:	c1 e7 02             	shl    $0x2,%edi
     d7d:	85 c0                	test   %eax,%eax
     d7f:	7e 54                	jle    dd5 <calcThreadColmajor+0xb5>
     d81:	83 e8 01             	sub    $0x1,%eax
     d84:	4d 89 f2             	mov    %r14,%r10
     d87:	4d 8d 64 c6 08       	lea    0x8(%r14,%rax,8),%r12
     d8c:	0f 1f 40 00          	nopl   0x0(%rax)
     d90:	8d 04 37             	lea    (%rdi,%rsi,1),%eax
     d93:	49 83 c2 08          	add    $0x8,%r10
     d97:	69 f6 8f bc 00 00    	imul   $0xbc8f,%esi,%esi
     d9d:	99                   	cltd   
     d9e:	f7 ff                	idiv   %edi
     da0:	48 63 d2             	movslq %edx,%rdx
     da3:	49 8b 04 d3          	mov    (%r11,%rdx,8),%rax
     da7:	49 89 42 f8          	mov    %rax,-0x8(%r10)
     dab:	8d 86 ff ff ff 7f    	lea    0x7fffffff(%rsi),%eax
     db1:	48 63 d0             	movslq %eax,%rdx
     db4:	48 89 d6             	mov    %rdx,%rsi
     db7:	48 c1 e6 1e          	shl    $0x1e,%rsi
     dbb:	48 01 d6             	add    %rdx,%rsi
     dbe:	99                   	cltd   
     dbf:	48 c1 fe 3d          	sar    $0x3d,%rsi
     dc3:	29 d6                	sub    %edx,%esi
     dc5:	89 f2                	mov    %esi,%edx
     dc7:	c1 e2 1f             	shl    $0x1f,%edx
     dca:	29 f2                	sub    %esi,%edx
     dcc:	29 d0                	sub    %edx,%eax
     dce:	4d 39 d4             	cmp    %r10,%r12
     dd1:	89 c6                	mov    %eax,%esi
     dd3:	75 bb                	jne    d90 <calcThreadColmajor+0x70>
     dd5:	44 89 c8             	mov    %r9d,%eax
     dd8:	89 35 46 12 20 00    	mov    %esi,0x201246(%rip)        # 202024 <seed.5154>
     dde:	4c 8b 7b 08          	mov    0x8(%rbx),%r15
     de2:	41 0f af c0          	imul   %r8d,%eax
     de6:	85 c0                	test   %eax,%eax
     de8:	7e 53                	jle    e3d <calcThreadColmajor+0x11d>
     dea:	83 e8 01             	sub    $0x1,%eax
     ded:	4d 89 fa             	mov    %r15,%r10
     df0:	4d 8d 64 c7 08       	lea    0x8(%r15,%rax,8),%r12
     df5:	0f 1f 00             	nopl   (%rax)
     df8:	8d 04 37             	lea    (%rdi,%rsi,1),%eax
     dfb:	49 83 c2 08          	add    $0x8,%r10
     dff:	69 f6 8f bc 00 00    	imul   $0xbc8f,%esi,%esi
     e05:	99                   	cltd   
     e06:	f7 ff                	idiv   %edi
     e08:	48 63 d2             	movslq %edx,%rdx
     e0b:	49 8b 04 d3          	mov    (%r11,%rdx,8),%rax
     e0f:	49 89 42 f8          	mov    %rax,-0x8(%r10)
     e13:	8d 86 ff ff ff 7f    	lea    0x7fffffff(%rsi),%eax
     e19:	48 63 d0             	movslq %eax,%rdx
     e1c:	48 89 d6             	mov    %rdx,%rsi
     e1f:	48 c1 e6 1e          	shl    $0x1e,%rsi
     e23:	48 01 d6             	add    %rdx,%rsi
     e26:	99                   	cltd   
     e27:	48 c1 fe 3d          	sar    $0x3d,%rsi
     e2b:	29 d6                	sub    %edx,%esi
     e2d:	89 f2                	mov    %esi,%edx
     e2f:	c1 e2 1f             	shl    $0x1f,%edx
     e32:	29 f2                	sub    %esi,%edx
     e34:	29 d0                	sub    %edx,%eax
     e36:	4d 39 d4             	cmp    %r10,%r12
     e39:	89 c6                	mov    %eax,%esi
     e3b:	75 bb                	jne    df8 <calcThreadColmajor+0xd8>
     e3d:	48 8b 43 10          	mov    0x10(%rbx),%rax
     e41:	48 c7 44 24 18 00 00 	movq   $0x3f800000,0x18(%rsp)
     e48:	80 3f 
     e4a:	ba 70 00 00 00       	mov    $0x70,%edx
     e4f:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
     e56:	00 00 
     e58:	41 50                	push   %r8
     e5a:	bf 65 00 00 00       	mov    $0x65,%edi
     e5f:	89 35 bf 11 20 00    	mov    %esi,0x2011bf(%rip)        # 202024 <seed.5154>
     e65:	be 6f 00 00 00       	mov    $0x6f,%esi
     e6a:	50                   	push   %rax
     e6b:	55                   	push   %rbp
     e6c:	41 51                	push   %r9
     e6e:	41 57                	push   %r15
     e70:	41 51                	push   %r9
     e72:	41 56                	push   %r14
     e74:	ff 74 24 40          	pushq  0x40(%rsp)
     e78:	e8 b3 fa ff ff       	callq  930 <cblas_cgemm@plt>
     e7d:	b8 cd cc cc cc       	mov    $0xcccccccd,%eax
     e82:	48 83 c4 40          	add    $0x40,%rsp
     e86:	41 f7 e5             	mul    %r13d
     e89:	c1 ea 03             	shr    $0x3,%edx
     e8c:	8d 04 92             	lea    (%rdx,%rdx,4),%eax
     e8f:	01 c0                	add    %eax,%eax
     e91:	41 39 c5             	cmp    %eax,%r13d
     e94:	74 32                	je     ec8 <calcThreadColmajor+0x1a8>
     e96:	41 83 c5 01          	add    $0x1,%r13d
     e9a:	41 83 fd 64          	cmp    $0x64,%r13d
     e9e:	0f 85 b4 fe ff ff    	jne    d58 <calcThreadColmajor+0x38>
     ea4:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
     ea9:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
     eb0:	00 00 
     eb2:	75 38                	jne    eec <calcThreadColmajor+0x1cc>
     eb4:	48 83 c4 38          	add    $0x38,%rsp
     eb8:	5b                   	pop    %rbx
     eb9:	5d                   	pop    %rbp
     eba:	41 5c                	pop    %r12
     ebc:	41 5d                	pop    %r13
     ebe:	41 5e                	pop    %r14
     ec0:	41 5f                	pop    %r15
     ec2:	c3                   	retq   
     ec3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     ec8:	48 8d 35 a5 03 00 00 	lea    0x3a5(%rip),%rsi        # 1274 <_IO_stdin_used+0x4>
     ecf:	44 89 ea             	mov    %r13d,%edx
     ed2:	bf 01 00 00 00       	mov    $0x1,%edi
     ed7:	31 c0                	xor    %eax,%eax
     ed9:	e8 12 fa ff ff       	callq  8f0 <__printf_chk@plt>
     ede:	48 8b 3d 33 11 20 00 	mov    0x201133(%rip),%rdi        # 202018 <stdout@@GLIBC_2.2.5>
     ee5:	e8 76 fa ff ff       	callq  960 <fflush@plt>
     eea:	eb aa                	jmp    e96 <calcThreadColmajor+0x176>
     eec:	e8 4f fa ff ff       	callq  940 <__stack_chk_fail@plt>
     ef1:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     ef6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     efd:	00 00 00 

0000000000000f00 <calcThreadRowmajor>:
     f00:	41 57                	push   %r15
     f02:	41 56                	push   %r14
     f04:	41 55                	push   %r13
     f06:	41 54                	push   %r12
     f08:	45 31 ed             	xor    %r13d,%r13d
     f0b:	55                   	push   %rbp
     f0c:	53                   	push   %rbx
     f0d:	48 89 fb             	mov    %rdi,%rbx
     f10:	48 83 ec 38          	sub    $0x38,%rsp
     f14:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
     f1b:	00 00 
     f1d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
     f22:	31 c0                	xor    %eax,%eax
     f24:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
     f29:	48 8d 6c 24 20       	lea    0x20(%rsp),%rbp
     f2e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
     f33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     f38:	8b 4b 24             	mov    0x24(%rbx),%ecx
     f3b:	44 8b 4b 2c          	mov    0x2c(%rbx),%r9d
     f3f:	8b 35 cb 10 20 00    	mov    0x2010cb(%rip),%esi        # 202010 <seed.5165>
     f45:	4c 8b 33             	mov    (%rbx),%r14
     f48:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
     f4c:	89 c8                	mov    %ecx,%eax
     f4e:	8d 3c 89             	lea    (%rcx,%rcx,4),%edi
     f51:	41 0f af c1          	imul   %r9d,%eax
     f55:	c1 e7 02             	shl    $0x2,%edi
     f58:	85 c0                	test   %eax,%eax
     f5a:	7e 59                	jle    fb5 <calcThreadRowmajor+0xb5>
     f5c:	83 e8 01             	sub    $0x1,%eax
     f5f:	4d 89 f0             	mov    %r14,%r8
     f62:	4d 8d 54 c6 08       	lea    0x8(%r14,%rax,8),%r10
     f67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
     f6e:	00 00 
     f70:	8d 04 37             	lea    (%rdi,%rsi,1),%eax
     f73:	49 83 c0 08          	add    $0x8,%r8
     f77:	69 f6 8f bc 00 00    	imul   $0xbc8f,%esi,%esi
     f7d:	99                   	cltd   
     f7e:	f7 ff                	idiv   %edi
     f80:	48 63 d2             	movslq %edx,%rdx
     f83:	49 8b 04 d3          	mov    (%r11,%rdx,8),%rax
     f87:	49 89 40 f8          	mov    %rax,-0x8(%r8)
     f8b:	8d 86 ff ff ff 7f    	lea    0x7fffffff(%rsi),%eax
     f91:	48 63 d0             	movslq %eax,%rdx
     f94:	48 89 d6             	mov    %rdx,%rsi
     f97:	48 c1 e6 1e          	shl    $0x1e,%rsi
     f9b:	48 01 d6             	add    %rdx,%rsi
     f9e:	99                   	cltd   
     f9f:	48 c1 fe 3d          	sar    $0x3d,%rsi
     fa3:	29 d6                	sub    %edx,%esi
     fa5:	89 f2                	mov    %esi,%edx
     fa7:	c1 e2 1f             	shl    $0x1f,%edx
     faa:	29 f2                	sub    %esi,%edx
     fac:	29 d0                	sub    %edx,%eax
     fae:	4d 39 c2             	cmp    %r8,%r10
     fb1:	89 c6                	mov    %eax,%esi
     fb3:	75 bb                	jne    f70 <calcThreadRowmajor+0x70>
     fb5:	44 8b 43 28          	mov    0x28(%rbx),%r8d
     fb9:	44 89 c8             	mov    %r9d,%eax
     fbc:	89 35 4e 10 20 00    	mov    %esi,0x20104e(%rip)        # 202010 <seed.5165>
     fc2:	4c 8b 7b 08          	mov    0x8(%rbx),%r15
     fc6:	41 0f af c0          	imul   %r8d,%eax
     fca:	85 c0                	test   %eax,%eax
     fcc:	7e 57                	jle    1025 <calcThreadRowmajor+0x125>
     fce:	83 e8 01             	sub    $0x1,%eax
     fd1:	4d 89 fa             	mov    %r15,%r10
     fd4:	4d 8d 64 c7 08       	lea    0x8(%r15,%rax,8),%r12
     fd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
     fe0:	8d 04 37             	lea    (%rdi,%rsi,1),%eax
     fe3:	49 83 c2 08          	add    $0x8,%r10
     fe7:	69 f6 8f bc 00 00    	imul   $0xbc8f,%esi,%esi
     fed:	99                   	cltd   
     fee:	f7 ff                	idiv   %edi
     ff0:	48 63 d2             	movslq %edx,%rdx
     ff3:	49 8b 04 d3          	mov    (%r11,%rdx,8),%rax
     ff7:	49 89 42 f8          	mov    %rax,-0x8(%r10)
     ffb:	8d 86 ff ff ff 7f    	lea    0x7fffffff(%rsi),%eax
    1001:	48 63 d0             	movslq %eax,%rdx
    1004:	48 89 d6             	mov    %rdx,%rsi
    1007:	48 c1 e6 1e          	shl    $0x1e,%rsi
    100b:	48 01 d6             	add    %rdx,%rsi
    100e:	99                   	cltd   
    100f:	48 c1 fe 3d          	sar    $0x3d,%rsi
    1013:	29 d6                	sub    %edx,%esi
    1015:	89 f2                	mov    %esi,%edx
    1017:	c1 e2 1f             	shl    $0x1f,%edx
    101a:	29 f2                	sub    %esi,%edx
    101c:	29 d0                	sub    %edx,%eax
    101e:	4d 39 d4             	cmp    %r10,%r12
    1021:	89 c6                	mov    %eax,%esi
    1023:	75 bb                	jne    fe0 <calcThreadRowmajor+0xe0>
    1025:	48 8b 43 10          	mov    0x10(%rbx),%rax
    1029:	48 c7 44 24 18 00 00 	movq   $0x3f800000,0x18(%rsp)
    1030:	80 3f 
    1032:	ba 6f 00 00 00       	mov    $0x6f,%edx
    1037:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    103e:	00 00 
    1040:	41 50                	push   %r8
    1042:	bf 65 00 00 00       	mov    $0x65,%edi
    1047:	89 35 c3 0f 20 00    	mov    %esi,0x200fc3(%rip)        # 202010 <seed.5165>
    104d:	be 6f 00 00 00       	mov    $0x6f,%esi
    1052:	50                   	push   %rax
    1053:	55                   	push   %rbp
    1054:	41 50                	push   %r8
    1056:	41 57                	push   %r15
    1058:	41 51                	push   %r9
    105a:	41 56                	push   %r14
    105c:	ff 74 24 40          	pushq  0x40(%rsp)
    1060:	e8 cb f8 ff ff       	callq  930 <cblas_cgemm@plt>
    1065:	b8 cd cc cc cc       	mov    $0xcccccccd,%eax
    106a:	48 83 c4 40          	add    $0x40,%rsp
    106e:	41 f7 e5             	mul    %r13d
    1071:	c1 ea 03             	shr    $0x3,%edx
    1074:	8d 04 92             	lea    (%rdx,%rdx,4),%eax
    1077:	01 c0                	add    %eax,%eax
    1079:	41 39 c5             	cmp    %eax,%r13d
    107c:	74 32                	je     10b0 <calcThreadRowmajor+0x1b0>
    107e:	41 83 c5 01          	add    $0x1,%r13d
    1082:	41 83 fd 64          	cmp    $0x64,%r13d
    1086:	0f 85 ac fe ff ff    	jne    f38 <calcThreadRowmajor+0x38>
    108c:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    1091:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
    1098:	00 00 
    109a:	75 38                	jne    10d4 <calcThreadRowmajor+0x1d4>
    109c:	48 83 c4 38          	add    $0x38,%rsp
    10a0:	5b                   	pop    %rbx
    10a1:	5d                   	pop    %rbp
    10a2:	41 5c                	pop    %r12
    10a4:	41 5d                	pop    %r13
    10a6:	41 5e                	pop    %r14
    10a8:	41 5f                	pop    %r15
    10aa:	c3                   	retq   
    10ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    10b0:	48 8d 35 d5 01 00 00 	lea    0x1d5(%rip),%rsi        # 128c <_IO_stdin_used+0x1c>
    10b7:	44 89 ea             	mov    %r13d,%edx
    10ba:	bf 01 00 00 00       	mov    $0x1,%edi
    10bf:	31 c0                	xor    %eax,%eax
    10c1:	e8 2a f8 ff ff       	callq  8f0 <__printf_chk@plt>
    10c6:	48 8b 3d 4b 0f 20 00 	mov    0x200f4b(%rip),%rdi        # 202018 <stdout@@GLIBC_2.2.5>
    10cd:	e8 8e f8 ff ff       	callq  960 <fflush@plt>
    10d2:	eb aa                	jmp    107e <calcThreadRowmajor+0x17e>
    10d4:	e8 67 f8 ff ff       	callq  940 <__stack_chk_fail@plt>
    10d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000010e0 <conplexUrandGenerator>:
    10e0:	85 c9                	test   %ecx,%ecx
    10e2:	41 89 d1             	mov    %edx,%r9d
    10e5:	7e 5a                	jle    1141 <conplexUrandGenerator+0x61>
    10e7:	8d 41 ff             	lea    -0x1(%rcx),%eax
    10ea:	48 8d 4c c6 08       	lea    0x8(%rsi,%rax,8),%rcx
    10ef:	90                   	nop
    10f0:	43 8d 04 01          	lea    (%r9,%r8,1),%eax
    10f4:	48 83 c6 08          	add    $0x8,%rsi
    10f8:	45 69 c0 8f bc 00 00 	imul   $0xbc8f,%r8d,%r8d
    10ff:	99                   	cltd   
    1100:	41 f7 f9             	idiv   %r9d
    1103:	48 63 d2             	movslq %edx,%rdx
    1106:	48 8b 04 d7          	mov    (%rdi,%rdx,8),%rax
    110a:	41 8d 90 ff ff ff 7f 	lea    0x7fffffff(%r8),%edx
    1111:	48 89 46 f8          	mov    %rax,-0x8(%rsi)
    1115:	48 63 c2             	movslq %edx,%rax
    1118:	49 89 c0             	mov    %rax,%r8
    111b:	49 c1 e0 1e          	shl    $0x1e,%r8
    111f:	49 01 c0             	add    %rax,%r8
    1122:	89 d0                	mov    %edx,%eax
    1124:	c1 f8 1f             	sar    $0x1f,%eax
    1127:	49 c1 f8 3d          	sar    $0x3d,%r8
    112b:	41 29 c0             	sub    %eax,%r8d
    112e:	44 89 c0             	mov    %r8d,%eax
    1131:	c1 e0 1f             	shl    $0x1f,%eax
    1134:	44 29 c0             	sub    %r8d,%eax
    1137:	29 c2                	sub    %eax,%edx
    1139:	48 39 f1             	cmp    %rsi,%rcx
    113c:	41 89 d0             	mov    %edx,%r8d
    113f:	75 af                	jne    10f0 <conplexUrandGenerator+0x10>
    1141:	44 89 c0             	mov    %r8d,%eax
    1144:	c3                   	retq   
    1145:	90                   	nop
    1146:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    114d:	00 00 00 

0000000000001150 <calcBcolmajor>:
    1150:	48 83 ec 18          	sub    $0x18,%rsp
    1154:	66 0f d6 44 24 08    	movq   %xmm0,0x8(%rsp)
    115a:	66 0f d6 0c 24       	movq   %xmm1,(%rsp)
    115f:	56                   	push   %rsi
    1160:	41 51                	push   %r9
    1162:	41 89 d1             	mov    %edx,%r9d
    1165:	48 8d 44 24 10       	lea    0x10(%rsp),%rax
    116a:	50                   	push   %rax
    116b:	52                   	push   %rdx
    116c:	41 50                	push   %r8
    116e:	52                   	push   %rdx
    116f:	41 89 f0             	mov    %esi,%r8d
    1172:	51                   	push   %rcx
    1173:	ba 70 00 00 00       	mov    $0x70,%edx
    1178:	89 f9                	mov    %edi,%ecx
    117a:	be 6f 00 00 00       	mov    $0x6f,%esi
    117f:	bf 65 00 00 00       	mov    $0x65,%edi
    1184:	48 8d 44 24 40       	lea    0x40(%rsp),%rax
    1189:	50                   	push   %rax
    118a:	e8 a1 f7 ff ff       	callq  930 <cblas_cgemm@plt>
    118f:	48 83 c4 58          	add    $0x58,%rsp
    1193:	c3                   	retq   
    1194:	66 90                	xchg   %ax,%ax
    1196:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    119d:	00 00 00 

00000000000011a0 <calcBrowmajor>:
    11a0:	48 83 ec 18          	sub    $0x18,%rsp
    11a4:	66 0f d6 44 24 08    	movq   %xmm0,0x8(%rsp)
    11aa:	66 0f d6 0c 24       	movq   %xmm1,(%rsp)
    11af:	56                   	push   %rsi
    11b0:	41 51                	push   %r9
    11b2:	41 89 d1             	mov    %edx,%r9d
    11b5:	48 8d 44 24 10       	lea    0x10(%rsp),%rax
    11ba:	50                   	push   %rax
    11bb:	56                   	push   %rsi
    11bc:	41 50                	push   %r8
    11be:	52                   	push   %rdx
    11bf:	41 89 f0             	mov    %esi,%r8d
    11c2:	51                   	push   %rcx
    11c3:	ba 6f 00 00 00       	mov    $0x6f,%edx
    11c8:	89 f9                	mov    %edi,%ecx
    11ca:	be 6f 00 00 00       	mov    $0x6f,%esi
    11cf:	bf 65 00 00 00       	mov    $0x65,%edi
    11d4:	48 8d 44 24 40       	lea    0x40(%rsp),%rax
    11d9:	50                   	push   %rax
    11da:	e8 51 f7 ff ff       	callq  930 <cblas_cgemm@plt>
    11df:	48 83 c4 58          	add    $0x58,%rsp
    11e3:	c3                   	retq   
    11e4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    11eb:	00 00 00 
    11ee:	66 90                	xchg   %ax,%ax

00000000000011f0 <__libc_csu_init>:
    11f0:	41 57                	push   %r15
    11f2:	41 56                	push   %r14
    11f4:	49 89 d7             	mov    %rdx,%r15
    11f7:	41 55                	push   %r13
    11f9:	41 54                	push   %r12
    11fb:	4c 8d 25 4e 0b 20 00 	lea    0x200b4e(%rip),%r12        # 201d50 <__frame_dummy_init_array_entry>
    1202:	55                   	push   %rbp
    1203:	48 8d 2d 4e 0b 20 00 	lea    0x200b4e(%rip),%rbp        # 201d58 <__init_array_end>
    120a:	53                   	push   %rbx
    120b:	41 89 fd             	mov    %edi,%r13d
    120e:	49 89 f6             	mov    %rsi,%r14
    1211:	4c 29 e5             	sub    %r12,%rbp
    1214:	48 83 ec 08          	sub    $0x8,%rsp
    1218:	48 c1 fd 03          	sar    $0x3,%rbp
    121c:	e8 7f f6 ff ff       	callq  8a0 <_init>
    1221:	48 85 ed             	test   %rbp,%rbp
    1224:	74 20                	je     1246 <__libc_csu_init+0x56>
    1226:	31 db                	xor    %ebx,%ebx
    1228:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    122f:	00 
    1230:	4c 89 fa             	mov    %r15,%rdx
    1233:	4c 89 f6             	mov    %r14,%rsi
    1236:	44 89 ef             	mov    %r13d,%edi
    1239:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
    123d:	48 83 c3 01          	add    $0x1,%rbx
    1241:	48 39 dd             	cmp    %rbx,%rbp
    1244:	75 ea                	jne    1230 <__libc_csu_init+0x40>
    1246:	48 83 c4 08          	add    $0x8,%rsp
    124a:	5b                   	pop    %rbx
    124b:	5d                   	pop    %rbp
    124c:	41 5c                	pop    %r12
    124e:	41 5d                	pop    %r13
    1250:	41 5e                	pop    %r14
    1252:	41 5f                	pop    %r15
    1254:	c3                   	retq   
    1255:	90                   	nop
    1256:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    125d:	00 00 00 

0000000000001260 <__libc_csu_fini>:
    1260:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000001264 <_fini>:
    1264:	48 83 ec 08          	sub    $0x8,%rsp
    1268:	48 83 c4 08          	add    $0x8,%rsp
    126c:	c3                   	retq   
