module tsvc_2_s323_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  real(c_double), allocatable, save :: s(:)
contains
  subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s323_fp64")
    real(c_double), intent(inout) :: a(*), b(*), c(*), d(*), e(*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    if (len_1d <= 1_c_int64_t) return
    if (.not. allocated(s) .or. size(s) < len_1d) then
       if (allocated(s)) deallocate(s)
       allocate(s(len_1d))
    end if
    ! Compute combined product s = c*d + c*e using two multiplies and an addition, in parallel
    !$omp parallel do schedule(static) default(none) shared(c,d,e,s,len_1d) private(i)
    do i = 2_c_int64_t, len_1d
      s(i) = c(i) * d(i) + c(i) * e(i)
    end do
    !$omp end parallel do
    ! Prefix sum of s (serial)
    do i = 3_c_int64_t, len_1d
      s(i) = s(i-1) + s(i)
    end do
    ! Compute final a and b using the precomputed sums and recomputed c*d product for a
    do i = 2_c_int64_t, len_1d
      if (i == 2_c_int64_t) then
        a(i) = b(1) + c(i) * d(i)
      else
        a(i) = b(1) + s(i-1) + c(i) * d(i)
      end if
      b(i) = b(1) + s(i)
    end do
  end subroutine tsvc_2_s323_fp64
end module tsvc_2_s323_mod