subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  integer(c_int64_t) :: i
  real(c_double) :: t0, t1, s1, s2, s3
  real(c_double) :: r1, r2, r3
  ! warmup + kernel work
  t0 = omp_get_wtime()
  if (len_1d > 1) then
    a(1) = b(1) * c(1)
!$omp parallel do simd schedule(static)
    do i = 2, len_1d
      a(i) = b(i) * c(i) + b(i - 1) * c(i - 1)
    end do
  else if (len_1d == 1) then
    a(1) = b(1) * c(1)
  end if
  t1 = omp_get_wtime()
  s1 = t1 - t0
  ! pure read of b and c (sum)
  r1 = 0.0d0
  t0 = omp_get_wtime()
!$omp parallel do reduction(+:r1) schedule(static)
  do i = 1, len_1d
    r1 = r1 + b(i) + c(i)
  end do
  t1 = omp_get_wtime()
  s2 = t1 - t0
  ! pure write of a
  t0 = omp_get_wtime()
!$omp parallel do schedule(static)
  do i = 1, len_1d
    a(i) = 0.25d0 * i
  end do
  t1 = omp_get_wtime()
  s3 = t1 - t0
  ! redo kernel so final output is correct
  if (len_1d > 1) then
    a(1) = b(1) * c(1)
!$omp parallel do simd schedule(static)
    do i = 2, len_1d
      a(i) = b(i) * c(i) + b(i - 1) * c(i - 1)
    end do
  end if
  write(6, '(A,F10.3,A,F10.3,A,F10.3,A,I0)') ' KERN_MS=', s1 * 1e3, ' READ_GBPS=', real(len_1d) * 16.0d0 / s2 / 1.0d6, ' WRITE_GBPS=', real(len_1d) * 8.0d0 / s3 / 1.0d6, ' NT=', omp_get_max_threads()
  flush(6)
end subroutine tsvc_2_s252_fp64
