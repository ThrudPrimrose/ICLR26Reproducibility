subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C, name='tsvc_2_s319_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), dimension(len_1d), target, intent(inout) :: a
  real(c_double), dimension(len_1d), target, intent(inout) :: b
  real(c_double), dimension(len_1d), target, intent(in)    :: c
  real(c_double), dimension(len_1d), target, intent(in)    :: d
  real(c_double), dimension(len_1d), target, intent(in)    :: e
  integer(c_int64_t) :: i
  real(c_double) :: t1, t2, s1, s2, sum

  t1 = 0.0d0
  t2 = 0.0d0
  s1 = 0.0d0
  s2 = 0.0d0
  if (len_1d >= 1048576) then
    !$omp parallel do reduction(+:s1) reduction(+:s2) num_threads(24)
    do i = 1, len_1d
      t1 = c(i) + d(i)
      t2 = c(i) + e(i)
      s1 = s1 + t1
      s2 = s2 + t2
      a(i) = t1
      b(i) = t2
    end do
    !$omp end parallel do
  else
    !$omp simd reduction(+:s1) reduction(+:s2)
    do i = 1, len_1d
      t1 = c(i) + d(i)
      t2 = c(i) + e(i)
      s1 = s1 + t1
      s2 = s2 + t2
      a(i) = t1
      b(i) = t2
    end do
    !$omp end simd
  end if
  sum = s1 + s2
  if (len_1d >= 1) b(1) = sum
end subroutine
