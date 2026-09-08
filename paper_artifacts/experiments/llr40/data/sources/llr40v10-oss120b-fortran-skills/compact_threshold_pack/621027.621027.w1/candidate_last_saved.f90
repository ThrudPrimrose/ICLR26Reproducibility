subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in) :: src(*)
  real(c_double), intent(in) :: weight(*)
  real(c_double), intent(inout) :: packed(*)
  integer(c_int64_t), intent(out) :: out_count
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t), pointer :: flag(:)
  integer(c_int64_t), pointer :: pos(:)
  integer(c_int64_t), pointer :: buffer(:)
  logical :: allocate_locally
  integer(c_int64_t) :: required_bytes
  integer(c_int64_t) :: i, total
  if (LEN_1D <= 0_c_int64_t) then
    out_count = 0_c_int64_t
    return
  end if
  required_bytes = 2_c_int64_t * LEN_1D * 8_c_int64_t
if (c_associated(workspace) .and. workspace_size >= required_bytes) then
    allocate_locally = .false.
    call c_f_pointer(workspace, buffer, [2*LEN_1D])
    flag => buffer(1:LEN_1D)
    pos => buffer(LEN_1D+1:2*LEN_1D)
else
    allocate_locally = .true.
    allocate(flag(LEN_1D))
    allocate(pos(LEN_1D))
end if
! Initialize packed to zero

  do i = 1, LEN_1D
    packed(i) = 0.0_c_double
  end do
  !$omp parallel do schedule(static) default(none) shared(src, flag, LEN_1D) private(i)
  do i = 1, LEN_1D
    if (src(i) > 0.0_c_double) then
      flag(i) = 1_c_int64_t
    else
      flag(i) = 0_c_int64_t
    end if
  end do
  !$omp end parallel do
  pos(1) = 0_c_int64_t
  do i = 2, LEN_1D
    pos(i) = pos(i-1) + flag(i-1)
  end do
  total = pos(LEN_1D) + flag(LEN_1D)
  !$omp parallel do schedule(static) default(none) shared(src, weight, packed, flag, pos, LEN_1D) private(i)
  do i = 1, LEN_1D
    if (flag(i) .eq. 1_c_int64_t) then
      packed(pos(i) + 1) = src(i) * weight(i)
    end if
  end do
  !$omp end parallel do
  out_count = total
  if (allocate_locally) then
    deallocate(flag)
    deallocate(pos)
  end if
end subroutine compact_threshold_pack_fp64
