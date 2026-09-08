subroutine tsvc_2_s1232_fp64(aa, bb, cc, len2d, vlen) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len2d, vlen
  real(c_double), intent(inout) :: aa(len2d, len2d)
  real(c_double), intent(in)    :: bb(len2d, len2d), cc(len2d, len2d)
  integer(c_int64_t) :: i, j, hi, lo, hi_i
  integer :: t, nt

  if (len2d <= 0) return
  if (vlen < 1) then
    !$omp parallel do schedule(static)
    do i = 1, len2d
      do j = 1, len2d
        aa(j, i) = bb(j, i) + cc(j, i)
      end do
    end do
  else
    nt = omp_get_max_threads()
    !$omp parallel default(none) shared(aa, bb, cc, len2d, vlen, nt) private(i, j, hi, lo, hi_i, t)
    t = omp_get_thread_num()
    lo   = len2d * int(sqrt(real(t, 8)   / real(nt, 8)))
    hi_i = len2d * int(sqrt(real(t + 1, 8) / real(nt, 8)))
    do i = lo + 1, hi_i
      hi = (i - 1) / vlen + 1
      do j = 1, hi
        aa(j, i) = bb(j, i) + cc(j, i)
      end do
    end do
    !$omp end parallel
  end if
end subroutine tsvc_2_s1232_fp64
