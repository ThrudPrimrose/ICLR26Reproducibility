subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in)    :: bb(len_2d, len_2d), cc(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j, i0, i1, chunk, nt
  real(c_double), volatile :: t

  nt = omp_get_max_threads()
  chunk = (len_2d + nt - 1) / nt
  if (chunk < 32) chunk = 32

  !$omp parallel do private(t)
  do i0 = 1, len_2d, chunk
    i1 = min(i0 + chunk - 1, len_2d)
    do j = 2, len_2d
      do i = i0, i1
        if (aa(i, 1) > 0.0d0) then
          t = bb(i, j) * cc(i, j)
          aa(i, j) = aa(i, j - 1) + t
        end if
      end do
    end do
  end do
end subroutine tsvc_2_s275_fp64
