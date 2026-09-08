subroutine tsvc_2_s3111_fp64(a, b, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(2)
  real(c_double), allocatable :: w(:)
  real(c_double) :: s, t0, t1
  integer(c_int64_t) :: i, n

  n = len_1d
  allocate(w(n))
  do i = 1, n
    w(i) = dble(mod(i, 7_8) - 3)
  end do

  ! P1: pure sum on harness a
  t0 = omp_get_wtime()
  s = 0.0d0
  !$omp parallel do simd reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
  t1 = omp_get_wtime()
  write(*, '(A,F10.6,A,F10.2)') 'P1_puresum_a t=', t1-t0, ' GB/s=', 8.0d0*dble(n)/1.0d9/max(t1-t0,1.0d-12)
  b(1) = s

  ! P2: merge sum on harness a (the real kernel)
  t0 = omp_get_wtime()
  s = 0.0d0
  !$omp parallel do simd reduction(+:s)
  do i = 1, n
    s = s + merge(a(i), 0.0d0, a(i) > 0.0d0)
  end do
  t1 = omp_get_wtime()
  write(*, '(A,F10.6,A,F10.2)') 'P2_merge_a   t=', t1-t0, ' GB/s=', 8.0d0*dble(n)/1.0d9/max(t1-t0,1.0d-12)

  ! P3: pure sum on local w
  t0 = omp_get_wtime()
  s = 0.0d0
  !$omp parallel do simd reduction(+:s)
  do i = 1, n
    s = s + w(i)
  end do
  t1 = omp_get_wtime()
  write(*, '(A,F10.6,A,F10.2)') 'P3_puresum_w t=', t1-t0, ' GB/s=', 8.0d0*dble(n)/1.0d9/max(t1-t0,1.0d-12)

  ! P4: merge sum on local w
  t0 = omp_get_wtime()
  s = 0.0d0
  !$omp parallel do simd reduction(+:s)
  do i = 1, n
    s = s + merge(w(i), 0.0d0, w(i) > 0.0d0)
  end do
  t1 = omp_get_wtime()
  write(*, '(A,F10.6,A,F10.2)') 'P4_merge_w   t=', t1-t0, ' GB/s=', 8.0d0*dble(n)/1.0d9/max(t1-t0,1.0d-12)
  b(2) = s
  write(*, '(A)') 'DONE'
  flush(6)
end subroutine tsvc_2_s3111_fp64
