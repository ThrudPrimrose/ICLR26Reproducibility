subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, LEN_1D) bind(C, name="compact_threshold_pack_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(in) :: src(*)
  real(c_double), intent(in) :: weight(*)
  real(c_double), intent(out) :: packed(*)
  integer(c_int64_t), intent(out) :: out_count
  ! real(c_double), allocatable :: prod(:)
  ! logical, allocatable :: mask(:)
  ! real(c_double), allocatable :: tmp(:)
  integer(c_int64_t) :: i, n
  external dummy

  if (LEN_1D <= 0_c_int64_t) then
    out_count = 0_c_int64_t
    return
  end if

  ! allocate(prod(LEN_1D))
  ! allocate(mask(LEN_1D))

  ! Compute packed values directly using PACK on expression
  n = 0_c_int64_t
  do i = 1, LEN_1D
    if (src(i) > 0.0_c_double) then
      call dummy()
      n = n + 1_c_int64_t
      packed(n) = src(i) * weight(i)
    end if
  end do
  out_count = n
  ! out_count = size(tmp, kind=c_int64_t)
  ! do i = 1, LEN_1D
    ! prod(i) = src(i) * weight(i)
    ! mask(i) = src(i) > 0.0_c_double
  ! end do
  ! Create mask of positive src values
  ! mask already set

  ! Pack the product values where mask is true
  ! tmp = pack(prod, mask)
! ! out_count = size(tmp, kind=c_int64_t)

  ! removed block

  ! deallocate(tmp)
end subroutine compact_threshold_pack_fp64

subroutine dummy()
  implicit none
end subroutine dummy
