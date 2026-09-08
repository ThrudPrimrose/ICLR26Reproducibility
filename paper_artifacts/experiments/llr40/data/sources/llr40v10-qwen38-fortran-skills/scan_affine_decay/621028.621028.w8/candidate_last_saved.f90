subroutine scan_affine_decay_fp64(y, c, x, n, ws, wssize) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: y(n)
  real(c_double), intent(in) :: c(n), x(n)
  type(c_ptr), value :: ws
  integer(c_int64_t), value, intent(in) :: wssize

  integer(c_int64_t) :: i, b, s, e, M, base, rem
  integer(c_int64_t) :: nt
  real(c_double) :: A, loc, prev, inA, inB
  real(c_double), allocatable :: Ab(:), Bb(:), inAb(:), inBb(:)
  integer :: tid

  if (n <= 0) return
  if (n == 1) then
     y(1) = x(1)
     return
  end if
  if (n < 4096) then
     y(1) = x(1)
     do i = 2, n
        y(i) = c(i)*y(i-1) + x(i)
     end do
     return
  end if

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  allocate(Ab(nt), Bb(nt), inAb(nt), inBb(nt))
  y(1) = x(1)
  M = n - 1
  base = M / nt
  rem  = M - base*nt
  print '(A,I0)', ' n = ', n
  print '(A,I0)', ' nt = ', nt
  print '(A,I0)', ' base = ', base
  print '(A,I0)', ' rem = ', rem

  !$omp parallel default(none) shared(c, x, y, Ab, Bb, inAb, inBb, M, n, nt, base, rem)
  !$omp private(b, s, e, i, A, loc, prev, inA, inB)
  b = omp_get_thread_num() + 1
  if (omp_get_thread_num() == 0) print '(A,I0,A,I0)', ' t0 s=', s, ' e=', e
  s = 2 + (b-1)*base + min(b-1, rem)
  e = 2 + b*base   + min(b,   rem) - 1
  
  A = 1.0d0
  loc = 0.0d0
  do i = s, e
     A = A * c(i)
     loc = c(i)*loc + x(i)
  end do
  Ab(b) = A
  Bb(b) = loc
  !$omp barrier
  !$omp single
  inA = 1.0d0; inB = 0.0d0
  do b = 1, nt
     inAb(b) = inA
     inBb(b) = inB
     inA = Ab(b) * inA
     inB = Ab(b) * inB + Bb(b)
  end do
  print '(A,ES15.6,A,ES15.6)', ' inAb(1)=', inAb(1), ' inBb(2)=', inBb(2)
  print '(A,ES15.6,A,ES15.6)', ' Ab(2)=', Ab(2), ' Bb(1)=', Bb(1)
  !$omp end single
  !$omp barrier
  b = omp_get_thread_num() + 1
  s = 2 + (b-1)*base + min(b-1, rem)
  e = 2 + b*base   + min(b,   rem) - 1
  prev = inAb(b)*y(1) + inBb(b)
  do i = s, e
     y(i) = c(i)*prev + x(i)
     prev = y(i)
  end do
  !$omp end parallel
  print '(A,ES15.6,A,ES15.6,A,ES15.6,A,ES15.6)', ' y1=', y(1), ' y2=', y(2), ' y3=', y(3), ' y10=', y(10)
  flush(6)
  deallocate(Ab, Bb, inAb, inBb)
end subroutine
